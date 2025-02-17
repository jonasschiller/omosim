package de.uniwuerzburg.omod.core

import com.graphhopper.GraphHopper
import com.graphhopper.gtfs.PtRouter
import de.uniwuerzburg.omod.core.models.*
import de.uniwuerzburg.omod.io.json.readJsonFromResource
import de.uniwuerzburg.omod.routing.*
import de.uniwuerzburg.omod.utils.ProgressBar
import de.uniwuerzburg.omod.utils.createCumDist
import de.uniwuerzburg.omod.utils.sampleCumDist
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import java.time.LocalDate
import java.time.LocalTime
import java.util.*
import java.util.concurrent.atomic.AtomicInteger
import kotlin.math.abs
import kotlin.math.exp
import kotlin.time.TimeSource

/**
 * Do mode choice with the mode options: FOOT, BICYCLE, CAR_DRIVER, CAR_PASSENGER, PUBLIC_TRANSIT
 * Uses a multinomial logit model at its core.
 *
 * The difference to ModeChoiceGTFS is that this does not use the travel time of each mode and takes the approximated
 * car distance from the routing cache. Making it significantly faster.
 *
 * @param routingCache GraphHopper for routing
 */
class ModeChoiceFast(
    private val routingCache: RoutingCache
) : ModeChoice {
    private val tourModeOptions: Array<ModeUtility> = readJsonFromResource("tourModeUtilities.json")
    private val tripModeOptions: Array<ModeUtility> = readJsonFromResource("tripModeUtilities.json")

    /**
     * Determine the mode of each trip and calculate the distance and time.
     *
     * @param agents Agents with trips (usually the trips have an UNDEFINED mode at this point)
     * @param mainRng Random number generator of the main thread
     * @param dispatcher Coroutine dispatcher used for concurrency
     * @return agents. Now their trips have specified modes.
     */
    override fun doModeChoice(
        agents: List<MobiAgent>, mainRng: Random, dispatcher: CoroutineDispatcher
    ) : List<MobiAgent> {
        val timeSource = TimeSource.Monotonic
        val timestampStartInit = timeSource.markNow()
        val jobsDone = AtomicInteger()
        val totalJobs = (agents.size).toDouble()

        for (chunk in agents.chunked(AppConstants.nAllowedCoroutines)) { // Don't launch to many coroutines at once
            runBlocking(dispatcher) {
                for (agent in chunk) {
                    val coroutineRng = Random(mainRng.nextLong())
                    launch(dispatcher) {
                        doModeChoiceFor(agent, coroutineRng)
                        val done = jobsDone.incrementAndGet()
                        print("Mode Choice: ${ProgressBar.show(done / totalJobs)}\r")
                    }
                }
            }
        }
        println("Mode Choice: " + ProgressBar.done())
        logger.get()?.info("Mode Choice took: ${timeSource.markNow() - timestampStartInit}")
        return agents
    }

    /**
     * Run through day and get all HOME-HOME tours. If the day does not start with a HOME activity,
     * all the trips before the first HOME activity are counted as a tour. The same is true for all the trips after the
     * last HOME activity if the day does not end with a HOME activity.
     *
     * @param diary Mobility pattern on a given day
     * @param rng Random number generator
     * @return Tours
     */
    private fun getTours(diary: Diary, rng: Random) : List<List<TripMCFeatures>> {
        val tours = mutableListOf<List<TripMCFeatures>>()
        var currentTour = mutableListOf<TripMCFeatures>()

        val visitor = {
            trip: Trip, originActivity: Activity, destinationActivity: Activity,
            departureTime: LocalTime, wd: Weekday, finished: Boolean ->

            // Routes for possible trips
            val rtDistances = mapOf(
                ActivityType.HOME to Route.sampleDistanceRoundTrip(ActivityType.HOME, rng),
                ActivityType.WORK to Route.sampleDistanceRoundTrip(ActivityType.WORK, rng),
                ActivityType.SCHOOL to Route.sampleDistanceRoundTrip(ActivityType.SCHOOL, rng)
            )

            // Get car distance and routes per mode
            val carDistance = if (
                (originActivity.type == destinationActivity.type) &&
                (
                    (originActivity.type == ActivityType.HOME) ||
                    (originActivity.type == ActivityType.WORK) ||
                    (originActivity.type == ActivityType.SCHOOL)
                )
            ) {
                // IF trip is from fixed location to same fixed location. Impute a randomly sampled Round-trip.
                rtDistances[originActivity.type]!!
              }  else {
                routingCache.getDistances(originActivity.location, listOf(destinationActivity.location))
                    .first()
                    .toDouble()
            }

            val tripFeatures = TripMCFeatures(
                carDistance,
                originActivity,
                destinationActivity,
                wd
            )
            currentTour.add(tripFeatures)

            // Tour ends
            if ((destinationActivity.type == ActivityType.HOME) || finished) {
                tours.add(currentTour)
                currentTour = mutableListOf()
            }

            // Add estimated travel time
            trip.time = null // TODO create second cache?
        }
        diary.visitTrips(visitor) // Run through day
        return tours
    }

    /**
     * Do Mode choice for a single Agent.
     * The result is directly stored in the agents diaries.
     *
     * @param agent Agent
     * @param rng Random number generator
     */
    private fun doModeChoiceFor(agent: MobiAgent, rng: Random) {
        for (diary in agent.mobilityDemand) {
            val tours = getTours(diary, rng)

            // Tour mode choice
            for (tour in tours) {
                // Only do HOME-HOME tours as one block
                if (tour.first().fromActivity.type != ActivityType.HOME) { continue }
                if (tour.last().toActivity.type != ActivityType.HOME) { continue }

                // Aggregate distance and times
                val carDistance = tour.sumOf { it.carDistance }

                // Main purpose of tour is defined by the activity with the longest stay time
                val mainPurpose = tour
                    .dropLast(1)
                    .maxByOrNull { it.toActivity.stayTime!! }?.toActivity?.type ?: ActivityType.HOME

                // Weekday
                val weekday = tour.first().weekday

                val mode = sampleUtilities(tourModeOptions, carDistance, agent, mainPurpose, weekday, rng)

                // If the tour is a CAR or BICYCLE all trips on the tour must be conducted with the respective vehicle
                if ((mode == Mode.CAR_DRIVER) || (mode == Mode.BICYCLE)) {
                    for (trip in tour) {
                        trip.mode = mode
                    }
                }
            }

            // Trip mode choice
            for (trip in tours.flatten().filter { it.mode == null }) {
                val mode = sampleUtilities(
                    tripModeOptions, trip.carDistance, agent, trip.toActivity.type, trip.weekday, rng
                )
                trip.mode = mode
            }

            // Format for output
            val outTrips = mutableListOf<Trip>()
            for (trip in tours.flatten()) {
                outTrips.add(
                    Trip(
                        trip.carDistance,
                        null,
                        mode = trip.mode!!,
                        lats = null,
                        lons = null
                    )
                )
            }
            diary.trips = outTrips
        }
    }

    /**
     * Sample logit model defined by the given utilities.
     *
     * @param options Possible modes for the decision (Different for tours and trips)
     * @param carDistance Distance by car. Used as the reference distance.
     * @param agent Agent
     * @param activity Main activity of the tour or purpose of the trip.
     * @param rng Random number generator
     * @return Chosen mode
     */
    private fun sampleUtilities(
        options: Array<ModeUtility>,
        carDistance: Double, agent: MobiAgent, activity: ActivityType,
        weekday: Weekday, rng: Random
    ) : Mode {
        // Sampling
        val weights = options
            .map { util -> exp(util.calc(null, carDistance, activity, agent.carAccess, weekday, agent)) }
            .toDoubleArray()
        val distr = createCumDist(weights)
        val mode = options[sampleCumDist(distr, rng)].mode
        return mode
    }

    /**
     * Utility data class that stores all features of a trip required by the logit model.
     * Also stores the result of the tour level decision for trip level decisions.
     *
     * @param carDistance Distance by car
     * @param fromActivity Activity before the trip
     * @param toActivity Activity after the trip
     */
    private class TripMCFeatures (
        val carDistance: Double,
        val fromActivity: Activity,
        val toActivity: Activity,
        val weekday: Weekday
    ) {
        var mode: Mode? = null
    }
}
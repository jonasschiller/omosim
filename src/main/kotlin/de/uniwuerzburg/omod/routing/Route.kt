package de.uniwuerzburg.omod.routing

import com.graphhopper.GHResponse
import com.graphhopper.GraphHopper
import com.graphhopper.gtfs.PtRouter
import de.uniwuerzburg.omod.core.models.ActivityType
import de.uniwuerzburg.omod.core.models.LocationOption
import de.uniwuerzburg.omod.core.models.Mode
import de.uniwuerzburg.omod.core.models.RealLocation
import de.uniwuerzburg.omod.utils.createCumDist
import de.uniwuerzburg.omod.utils.sampleCumDist
import java.time.Instant
import java.util.*
import kotlin.math.ln

class Route (
    val distance: Double,           // Unit: kilometer
    val time: Double,               // Unit: minutes
    val lats: List<Double>?,
    val lons: List<Double>?,
    val onlyWalk: Boolean = false
) {
    companion object {
        fun getWithFallbackAlt(
            mode: Mode, origin: LocationOption, destination: LocationOption, hopper: GraphHopper, withPath: Boolean,
            departureTime: Instant?, ptRouter: PtRouter?,
            rng: Random,
            altPercentages: Map<Pair<RealLocation, RealLocation>, List<Double>>? = null
        ): Route {
            // TODO test implementation
            if (((departureTime == null) || (ptRouter == null)) && (mode == Mode.PUBLIC_TRANSIT)) {
                throw IllegalArgumentException("A ptRouter and departureTime is required for public transit routing!")
            }

            val route = if ((origin !is RealLocation) || (destination !is RealLocation)) {
                routeFallback(mode, origin, destination)
            } else {
                var alts = altPercentages?.get(
                    Pair( origin.getAggLoc()!! as RealLocation, destination.getAggLoc()!! as RealLocation)
                )

                val response = when (mode) {
                    Mode.PUBLIC_TRANSIT -> { routeGTFS(origin, destination, departureTime!!, ptRouter!!, hopper) }
                    Mode.FOOT           -> routeWith("foot", origin, destination, hopper)
                    Mode.BICYCLE        -> routeWith("bike", origin, destination, hopper)
                    else                -> {
                        if (alts == null) {
                            routeWith("car", origin, destination, hopper)
                        } else {
                            // TODO buildings not cells
                            routeAltCar(
                                origin.getAggLoc()!! as RealLocation, destination.getAggLoc()!! as RealLocation, hopper
                            )
                        }
                    }
                }
                if (response.hasErrors()) {
                    routeFallback(mode, origin, destination)
                } else if ((mode == Mode.CAR_DRIVER) && (alts != null)){
                    fromGHAltResponse(response, withPath, alts, rng)
                } else {
                    fromGHResponse(response, withPath)
                }
            }
            return addConstantTimeCost(mode, route)
        }

        fun getWithFallback(
            mode: Mode, origin: LocationOption, destination: LocationOption, hopper: GraphHopper, withPath: Boolean,
            departureTime: Instant?, ptRouter: PtRouter?
        ): Route {
            if (((departureTime == null) || (ptRouter == null)) && (mode == Mode.PUBLIC_TRANSIT)) {
                throw IllegalArgumentException("A ptRouter and departureTime is required for public transit routing!")
            }

            val route = if ((origin !is RealLocation) || (destination !is RealLocation)) {
                routeFallback(mode, origin, destination)
            } else {
                val response = when (mode) {
                    Mode.PUBLIC_TRANSIT -> { routeGTFS(origin, destination, departureTime!!, ptRouter!!, hopper) }
                    Mode.FOOT           -> routeWith("foot", origin, destination, hopper)
                    Mode.BICYCLE        -> routeWith("bike", origin, destination, hopper)
                    else                -> routeWith("car", origin, destination, hopper)
                }
                if (response.hasErrors()) {
                    routeFallback(mode, origin, destination)
                } else {
                    fromGHResponse(response, withPath)
                }
            }
            return addConstantTimeCost(mode, route)
        }

        private fun fromGHAltResponse(
            response: GHResponse, withPath: Boolean, probs: List<Double>, rng: Random
        ) : Route {
            require(response.all.size == probs.size)
            val distr = createCumDist(probs.toDoubleArray())
            val path = response.all[sampleCumDist(distr, rng)]

            val (lats, lons) = if (withPath) {
                Pair(
                    path.points.map { it.lat },
                    path.points.map { it.lon }
                )
            } else {
                Pair(null, null)
            }

            // Check if only walking occurred
            val onlyWalk = response.best.legs.all { it.type == "walk" }

            return Route (
                response.best.distance / 1000,
                (response.best.time / 1000 / 60).toDouble(),
                lats,
                lons,
                onlyWalk
            )
        }

        private fun fromGHResponse(response: GHResponse, withPath: Boolean) : Route {
            val (lats, lons) = if (withPath) {
                Pair(
                    response.best.points.map { it.lat },
                    response.best.points.map { it.lon }
                )
            } else {
                Pair(null, null)
            }

            // Check if only walking occurred
            val onlyWalk = response.best.legs.all { it.type == "walk" }

            return Route (
                response.best.distance / 1000,
                (response.best.time / 1000 / 60).toDouble(),
                lats,
                lons,
                onlyWalk
            )
        }

        private fun routeFallback(mode: Mode, origin: LocationOption, destination: LocationOption): Route {
            val beelineDistance = calcDistanceBeeline(origin, destination) / 1000
            return routeFallbackFromDistance(mode, beelineDistance)
        }

        /**
         * @param mode Travel mode
         * @param distance Distance of trip. Unit: Kilometer
         */
        private fun routeFallbackFromDistance(mode: Mode, distance: Double): Route {
            val time = when (mode) {
                Mode.PUBLIC_TRANSIT -> distance / 22.5 * 60 // 22.5 km/h
                Mode.FOOT -> distance / 5 * 60 // 5 km/h
                Mode.BICYCLE -> distance / 18 * 60 // 18 km/h
                else -> distance / 75  * 60 // 75 km/h + 5 min for Parking
            }
            return Route(distance, time,null, null)
        }

        fun sampleDistanceRoundTrip(activityType: ActivityType, rng: Random): Double {
            val lambda: Double = when (activityType) {
                ActivityType.HOME   -> 1.0 / 11.9
                ActivityType.WORK   -> 1.0 / 12.7
                ActivityType.SCHOOL -> 1.0 / 7.7
                else -> throw IllegalArgumentException(
                    "Round-trips must start and end at a fixed location (HOME, WORK, SCHOOL)."
                )
            }
            return 1 / (-lambda) * ln(1-rng.nextDouble())
        }

        fun getRoundTripRoute(mode: Mode, distance: Double): Route  {
            val route = routeFallbackFromDistance(mode, distance)
            return addConstantTimeCost(mode, route)
        }

        private fun addConstantTimeCost(mode: Mode, route: Route) : Route {
            return when (mode) {
                Mode.CAR_DRIVER    -> addTimeAndCopy(5, route) // 5min for parking
                Mode.CAR_PASSENGER -> addTimeAndCopy(5, route) // 5min for parking
                Mode.BICYCLE       -> addTimeAndCopy(1, route) // 1min locking/unlocking and parking
                else               -> route
            }
        }

        private fun addTimeAndCopy(time: Int, route: Route) : Route {
            return Route(route.distance, route.time + time, route.lats, route.lons)
        }
    }
}
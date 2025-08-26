package de.uniwuerzburg.omod.core

import de.uniwuerzburg.omod.calibration.differentiablemodel.LinearBaseTerm
import de.uniwuerzburg.omod.calibration.differentiablemodel.Term
import de.uniwuerzburg.omod.core.models.*
import kotlinx.serialization.Serializable
import kotlin.math.ln
import kotlin.math.max

/**
 * Utility formula for a mode in a mode choice logit model.
 */
@Serializable
data class ModeUtility (
    val mode: Mode,
    val timeCoeff: Double,
    val logTimeCoeff: Double,
    val distanceCoeff: Double,
    val logDistanceCoeff: Double,
    val homGroupCoeff: Map<HomogeneousGrp, Double>,
    val mobGroupCoeff: Map<MobilityGrp, Double>,
    val ageGrpCoeff: Map<AgeGrp, Double>,
    val weekdayCoeff: Map<Weekday, Double>,
    val sexCoeff: Map<Sex, Double>,
    val carAvailableCoeff: Map<Boolean, Double>,
    val activityCoeff: Map<ActivityType, Double>,
    val intercept: Double
) {
    /**
     * Calculate utility.
     *
     * @param time Travel time of the trip/tour with that mode.
     * @param distance Reference distance of the trip/tour (Usually car distance)
     * @param activity Main activity of tour or purpose of trip.
     * @param carAvailable Is a car available at time the decision is made.
     * @param agent Agent
     * @return Utility
     */
    fun calc(
        time: Double?, distance: Double, activity: ActivityType, carAvailable: Boolean?, weekday: Weekday,
        agent: MobiAgent
    ) : Double {
        val (timeClipped, lnTimeClipped) = if (time != null) {
            val t = max(time, 1.0) // Minimum time: 1 minute
            val lnt = ln(t)
            t to lnt
        } else {
            val t = 0.0
            val lnt = 0.0
            t to lnt
        }

        val distanceClipped = max(distance, 0.001) // Minimum distance: 1 meter

        return timeClipped * timeCoeff +
               lnTimeClipped * logTimeCoeff +
               distanceClipped * distanceCoeff +
               ln(distanceClipped) * logDistanceCoeff +
               (homGroupCoeff[agent.homogenousGroup] ?: 0.0) +
               (mobGroupCoeff[agent.mobilityGroup] ?: 0.0) +
               (ageGrpCoeff[agent.ageGrp] ?: 0.0) +
               (weekdayCoeff[weekday] ?: 0.0) +
               (sexCoeff[agent.sex] ?: 0.0) +
               (carAvailableCoeff[carAvailable] ?: 0.0) +
               (activityCoeff[activity] ?: 0.0) +
               intercept
    }

    fun calTerm(
        time: Double?, distance: Double, activity: ActivityType, carAvailable: Boolean?, weekday: Weekday,
        agent: MobiAgent
    ) : Term {
        val (timeClipped, lnTimeClipped) = if (time != null) {
            val t = max(time, 1.0) // Minimum time: 1 minute
            val lnt = ln(t)
            t to lnt
        } else {
            val t = 0.0
            val lnt = 0.0
            t to lnt
        }

        val varActivity = when(activity) {
            ActivityType.HOME -> 2
            ActivityType.SHOPPING -> 3
            ActivityType.SCHOOL -> 4
            ActivityType.WORK -> 5
            else -> -1
        }

        val distanceClipped = max(distance, 0.001) // Minimum distance: 1 meter
        val cnst = (homGroupCoeff[agent.homogenousGroup] ?: 0.0) +
                (mobGroupCoeff[agent.mobilityGroup] ?: 0.0) +
                (ageGrpCoeff[agent.ageGrp] ?: 0.0) +
                (weekdayCoeff[weekday] ?: 0.0) +
                (sexCoeff[agent.sex] ?: 0.0) +
                (carAvailableCoeff[carAvailable] ?: 0.0) +
                (activityCoeff[activity] ?: 0.0)

        val term = LinearBaseTerm(6)
        term.addConstant(cnst)
        //term.addTerm(0, timeClipped)
        //term.addTerm(0, lnTimeClipped)
        //term.addTerm(2, distanceClipped)
        term.addTerm(0,  ln(distanceClipped))
        term.addTerm(1,  1.0)
        if (varActivity != -1) {
            term.addTerm(varActivity,  1.0)
        }
        return term
    }
}
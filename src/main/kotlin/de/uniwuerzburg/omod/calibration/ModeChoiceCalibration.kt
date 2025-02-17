package de.uniwuerzburg.omod.calibration

import de.uniwuerzburg.omod.core.ModeUtility
import de.uniwuerzburg.omod.core.models.ActivityType
import de.uniwuerzburg.omod.core.models.MobiAgent
import de.uniwuerzburg.omod.core.models.Weekday
import de.uniwuerzburg.omod.io.json.readJsonFromResource
import kotlin.math.exp

class ModeChoiceCalibration {
    private val tripModeOptions: Array<ModeUtility> = readJsonFromResource("tripModeUtilitiesCalibration.json")

    fun utilitiesForCalibration(
        carDistance: Double, agent: MobiAgent, activity: ActivityType, weekday: Weekday
    ) : DoubleArray {
        val weights = tripModeOptions
            .map { util -> exp(util.calc(null, carDistance, activity, agent.carAccess, weekday, agent)) }
            .toDoubleArray()
        return weights
    }
}
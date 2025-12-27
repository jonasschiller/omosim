package de.uniwuerzburg.omosim.calibration

import de.uniwuerzburg.omosim.calibration.CalibrationConstants.T
import java.time.LocalTime
import kotlin.math.floor

/**
 * Determine the simulation time slice for a given LocalTime.
 * Accuracy: Minute
 */
fun LocalTime.determineTimeSlice() : Int {
    val mod = this.minute + this.hour * 60
    return floor((mod % 1440.0) / 1440.0 * T).toInt()
}
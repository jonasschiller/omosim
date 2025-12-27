package de.uniwuerzburg.omosim.cli

import de.uniwuerzburg.omosim.calibration.CalibrationOption
import de.uniwuerzburg.omosim.calibration.CalibrationType
import de.uniwuerzburg.omosim.core.models.ActivityType

class CalibrationStep (
    val type: CalibrationType?,
    val alg: CalibrationOption?,
    val maxIterations: Int?,
    val activityType: ActivityType?,
    val parameters: Map<String, String>
) {
    // TODO warn about ModeChoice before GRAVITY
    companion object {
        /**
         * Format = TYPE:ALG:ITERATIONS:ACTIVITY?:PARAMS
         */
        fun fromCLIString(str: String) : CalibrationStep {
            val components = str.split(":")
            require(components.size >= 5) {
                "Calibration step string must contain 4 sections. " +
                        "If you want to use the default value for a section leave the section empty." +
                        "Format: TYPE:ALG:ITERATIONS:ACTIVITY?:PARAMS"
            }

            // Parse calibration steps
            val type = if (components[0] == "") {
                null
            } else {
                try {
                    CalibrationType.valueOf(components[0])
                } catch (e: IllegalArgumentException) {
                    throw IllegalArgumentException()
                }

            }
            val alg = if (components[1] == "") {
                null
            } else {
                CalibrationOption.valueOf(components[1])
            }
            val maxIterations = components[2].toIntOrNull()
            val activityType = if (components[3] == "") {
                null
            } else {
                ActivityType.valueOf(components[1])
            }

            val parameters = mutableMapOf<String, String>()
            for (i in 4 until components.size) {
                val kv = components[i].split("=")
                require(kv.size == 2) {
                    "Parameter ${components[i]} incorrectly formatted. Format ParameterName=Value"
                }
                parameters[kv[0]] = kv[1]
            }

            return CalibrationStep(type, alg, maxIterations, activityType, parameters)
        }
    }
}
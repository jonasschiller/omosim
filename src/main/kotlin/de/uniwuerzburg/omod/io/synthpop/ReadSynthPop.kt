package de.uniwuerzburg.omod.io.synthpop

import de.uniwuerzburg.omod.core.models.HomogeneousGrp
import de.uniwuerzburg.omod.core.models.MobilityGrp
import de.uniwuerzburg.omod.core.models.Sex
import java.io.File
import kotlin.random.Random

fun getSynthPopAgents(filePath: String): List<SynthPopAgent> {
    return File(filePath)
        .bufferedReader()
        .useLines { lines ->
            lines.drop(1) // skip header
                .filter { it.isNotBlank() }
                .map { line ->
                    val parts = line.split(',', ignoreCase = false, limit = 14)

                    // Parse and map fields directly into the final SynthPopAgent
                    SynthPopAgent(
                        id = 0,
                        homogenousGroup = when (parts[8].trim().toInt()) {
                            in 1..11 -> HomogeneousGrp.WORKING
                            97 -> HomogeneousGrp.NON_WORKING
                            else -> HomogeneousGrp.UNDEFINED
                        },
                        mobilityGroup = when {
                            parts[5].trim().toInt() == 2 -> MobilityGrp.NOT_CAR // driverLicense
                            parts[6].trim().toInt() == 2 && parts[10].trim().toInt() >= 1 -> MobilityGrp.CAR_MIXED // driveRegularly + carGroup
                            parts[6].trim().toInt() == 2 && parts[10].trim().toInt() == 0 -> MobilityGrp.NOT_CAR
                            parts[6].trim().toInt() == 1 && parts[10].trim().toInt() >= 1 -> MobilityGrp.CAR_USER
                            parts[10].trim().toInt() == 0 -> MobilityGrp.CAR_MIXED
                            else -> MobilityGrp.UNDEFINED
                        },
                        age = parts[3].trim().toInt(),
                        sex = if (parts[2].trim().toInt() == 1) Sex.MALE else Sex.FEMALE,
                        carAccess = parts[10].trim().toInt() > 0,
                        homeOfficeDays = parts[9].trim().toInt(),
                        sharedOfficeRate = Random.nextDouble(),
                        DRTprobability = 0.0
                    )
                }.toList()
        }
}


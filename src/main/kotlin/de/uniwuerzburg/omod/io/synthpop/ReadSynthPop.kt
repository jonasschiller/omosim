package de.uniwuerzburg.omod.io.synthpop

import de.uniwuerzburg.omod.core.models.HomogeneousGrp
import de.uniwuerzburg.omod.core.models.MobilityGrp
import de.uniwuerzburg.omod.core.models.Sex
import org.locationtech.jts.geom.Coordinate
import java.io.File
import kotlin.random.Random
import kotlin.math.*

fun getSynthPopAgents(synthPopFile: File): List<SynthPopAgent> {
    return synthPopFile
        .bufferedReader()
        .useLines { lines ->
            lines.drop(1) // skip header
                .filter { it.isNotBlank() }
                .map { line ->
                    val parts = line.split(',')

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
                        age = when(parts[3].trim().toInt()) {
                            1 -> sampleIntInclusive(18,24)
                            2 -> sampleIntInclusive(25,34)
                            3 -> sampleIntInclusive(35,44)
                            4 -> sampleIntInclusive(45, 54)
                            5 -> sampleIntInclusive(55,64)
                            6 -> sampleAge65Plus() // sample age for 65+ using chi-square(df=2)
                            else -> 0
                        },
                        sex = if (parts[2].trim().toInt() == 1) Sex.MALE else Sex.FEMALE,
                        homeOfficeDays = when (parts[9].trim().toInt()) {
                            5 -> 0.0 // no home office
                            4 -> 1.5 // 1-2 days
                            3 -> 3.5 // 3 days
                            2, 1 -> 5.0 // 5 days or full home office
                            else -> 0.0
                        },
                        carAccess = parts[10].trim().toInt() > 0,
                        sharedOfficeLike = parts[14].trim().toInt(),
                        sharedOfficeDays = parts[15].trim().toInt(),
                        drtLike = parts[16].trim().toInt(),
                        sharedOfficeLocation = if (parts.size > 18) {
                                Coordinate(parts[17].trim().toDouble(), parts[18].trim().toDouble())
                            } else {
                                null
                            }
                    )
                        }.toList()
                }
        }

fun sampleIntInclusive(min: Int, max: Int, rnd: Random = Random.Default): Int {
    require(min <= max) { "min must be <= max" }
    // sichere Long-Variante, vermeidet overflow bei max == Int.MAX_VALUE
    return rnd.nextLong(min.toLong(), max.toLong() + 1L).toInt()
}

/**
 * Sample a chi-square random variable with df = 2.
 * Implementation: sum of squares of two independent standard normals (Box-Muller).
 * Returns a Double >= 0.
 */
fun sampleChiSquare2(rnd: Random = Random.Default): Double {
    // Box-Muller to get two independent standard normals
    val u1 = rnd.nextDouble() // in [0.0, 1.0)
    val u2 = rnd.nextDouble()
    // avoid log(0) by ensuring u1 > 0 (nextDouble() returns >=0.0 and <1.0; if u1 == 0 we bump it)
    val uu1 = if (u1 == 0.0) Double.MIN_VALUE else u1
    val r = sqrt(-2.0 * ln(uu1))
    val theta = 2.0 * PI * u2
    val z0 = r * cos(theta)
    val z1 = r * sin(theta)
    return z0 * z0 + z1 * z1
}

/**
 * Sample an integer age for the 65+ group using a chi-square(df=2) distribution.
 * We interpret the chi-square draw as an offset (>=0) added to 65. The offset is floored to an Int.
 * The result is coerced into the inclusive range [65, maxAge].
 */
fun sampleAge65Plus(maxAge: Int = 100, rnd: Random = Random.Default): Int {
    require(maxAge >= 65) { "maxAge must be >= 65" }
    val chi = sampleChiSquare2(rnd)
    val offset = floor(chi).toInt().coerceAtLeast(0)
    return (65 + offset).coerceAtMost(maxAge)
}

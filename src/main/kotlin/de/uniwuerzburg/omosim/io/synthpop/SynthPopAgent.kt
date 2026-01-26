package de.uniwuerzburg.omod.io.synthpop

import de.uniwuerzburg.omod.core.models.HomogeneousGrp
import de.uniwuerzburg.omod.core.models.MobilityGrp
import de.uniwuerzburg.omod.core.models.Sex
import org.locationtech.jts.geom.Coordinate

data class SynthPopAgent (
    val id: Int,
    val homogenousGroup: HomogeneousGrp,
    val mobilityGroup: MobilityGrp,
    val age: Int?,
    val homeOfficeDays: Double,
    val sharedOfficeLike: Int,
    val sharedOfficeDays: Int,
    val drtLike: Int,
    val sharedOfficeLocation: Coordinate? = null,
    val sex: Sex,
    var carAccess: Boolean = false,
    )
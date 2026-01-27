package de.uniwuerzburg.omosim.io.synthpop

import de.uniwuerzburg.omosim.core.models.HomogeneousGrp
import de.uniwuerzburg.omosim.core.models.MobilityGrp
import de.uniwuerzburg.omosim.core.models.Sex
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
package de.uniwuerzburg.omod.io.synthpop

import de.uniwuerzburg.omod.core.models.Diary
import de.uniwuerzburg.omod.core.models.HomogeneousGrp
import de.uniwuerzburg.omod.core.models.MobilityGrp
import de.uniwuerzburg.omod.core.models.Sex

data class SynthPopAgent (
    val id: Int,
    val homogenousGroup: HomogeneousGrp,
    val mobilityGroup: MobilityGrp,
    val age: Int?,
    val homeOfficeDays: Int,
    val sharedOfficeRate: Double,
    val DRTprobability: Double,
    val sex: Sex,
    var carAccess: Boolean = false,
    val mobilityDemand: MutableList<Diary> = mutableListOf(),
    )
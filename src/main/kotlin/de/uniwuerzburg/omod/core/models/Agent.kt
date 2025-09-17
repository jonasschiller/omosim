package de.uniwuerzburg.omod.core.models

sealed interface Agent{
    val id: Int
    val homogenousGroup: HomogeneousGrp
    val mobilityGroup: MobilityGrp
    val age: Int?
    val home: LocationOption
    val work: LocationOption
    val school: LocationOption
    val sex: Sex
    var carAccess: Boolean
    val mobilityDemand: MutableList<Diary>}

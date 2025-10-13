package de.uniwuerzburg.omod.core.models

/**
 * Agent.
 *
 * @param id ID
 * @param homogenousGroup Hom. group of agent (Working person etc.)
 * @param mobilityGroup Mob. group of agent (Car user etc.)
 * @param age Age group of agent
 * @param home Home location of agent
 * @param work Work location of agent (Is also defined if the agent does not work)
 * @param school School location of agent (Is also defined if the agent does not go to school)
 * @param sex Sex of agent
 * @param mobilityDemand Simulation result of agent
 *
 */
abstract class MobiAgent(
    open val id: Int,
    open val homogenousGroup: HomogeneousGrp,
    open val mobilityGroup: MobilityGrp,
    open val age: Int?,
    open val home: LocationOption,
    open val work: LocationOption,
    open val shared_office: LocationOption?,
    open val school: LocationOption,
    open val sex: Sex,
    open var carAccess: Boolean = false,
    open val mobilityDemand: MutableList<Diary> = mutableListOf()
) {
    val ageGrp = AgeGrp.fromInt(age)
}

data class MobiAgentBase(
    override val id: Int,
    override val homogenousGroup: HomogeneousGrp,
    override val mobilityGroup: MobilityGrp,
    override val age: Int?,
    override val home: LocationOption,
    override val shared_office: LocationOption?,
    override val work: LocationOption,
    override val school: LocationOption,
    override val sex: Sex,
    override var carAccess: Boolean = false,
    override val mobilityDemand: MutableList<Diary> = mutableListOf()
) : MobiAgent(
    id, homogenousGroup, mobilityGroup, age, home, work, shared_office, school,sex, carAccess, mobilityDemand
)

data class MobiAgentSSWC(
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


package de.uniwuerzburg.omosim.calibration

class RouteChoiceCalibrationStore (
    val areaCacheID: String
) {
    companion object {
        fun write(
            altPercentages: Map<ODTTriple, List<Double>> = mapOf()
        ) {

        }

        fun read(areaCacheID: String) : Map<ODTTriple, List<Double>> {
            return  mapOf()
        }
    }
}
@file:Suppress("PropertyName")

package de.uniwuerzburg.omod

import com.github.ajalt.clikt.core.CliktCommand
import com.github.ajalt.clikt.parameters.arguments.argument
import com.github.ajalt.clikt.parameters.groups.*
import com.github.ajalt.clikt.parameters.options.*
import com.github.ajalt.clikt.parameters.types.*
import de.uniwuerzburg.omod.calibration.CalibrationInfo
import de.uniwuerzburg.omod.calibration.CalibrationOption
import de.uniwuerzburg.omod.calibration.TrafficCountCalibrator
import de.uniwuerzburg.omod.core.DestinationFinderDefault
import de.uniwuerzburg.omod.core.Omod
import de.uniwuerzburg.omod.core.logger
import de.uniwuerzburg.omod.core.models.ActivityType
import de.uniwuerzburg.omod.core.models.ModeChoiceOption
import de.uniwuerzburg.omod.core.models.Weekday
import de.uniwuerzburg.omod.io.formatOutput
import de.uniwuerzburg.omod.io.matsim.writeMatSim
import de.uniwuerzburg.omod.io.sqlite.writeSQLite
import de.uniwuerzburg.omod.routing.RoutingMode
import kotlinx.serialization.ExperimentalSerializationApi
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.encodeToStream
import java.io.File
import java.io.FileOutputStream
import java.nio.file.Paths

sealed interface AgentNumberDefinition

class FixedAgentNumber(
    val value: Int
) : AgentNumberDefinition
class ShareOfPop (
    val value: Double
) : AgentNumberDefinition

class CalibrationOptions : OptionGroup (
    help = "Calibration parameters."
) {
    val cal_traffic_count_file by option(
        help = "Traffic count data that serves as ground truth."
    ).file(mustExist = true, mustBeReadable = true).required()
    val cal_method by option(
        help = "Calibration algorithm to use."
    ).enum<CalibrationOption>().default(CalibrationOption.PSO)
    val cal_out by option(
        help = "Calibration output file. Stores the result of a calibration run"
    ).file().default(File("omosim_calibration.json"))
    val cal_activity by option(
        help = ""
    ).enum<ActivityType>().multiple()
    val cal_iterations by option(
        help = ""
    ).int().default(1000)
    val cal_parameter by option(
        help = ""
    ).splitPair()
     .multiple()
     .toMap()
}

/**
 * CLI interface
 */
@Suppress("PrivatePropertyName")
class Run : CliktCommand() {
    // Arguments
    private val area_geojson by argument(
        help = "Path to the GeoJSON file that defines the area for which you want to generate mobility demand. " +
                "Helpful websites for GeoJSON generation: https://geojson.io, https://polygons.openstreetmap.fr"
    ).file(mustExist = true, mustBeReadable = true)
    private val osm_file by argument(
        help = "Path to an osm.pbf file that covers the area completely. " +
               "Recommended download platform: https://download.geofabrik.de/"
    ).file(mustExist = true, mustBeReadable = true)
    // Options
    private val agentNumberDefinition by mutuallyExclusiveOptions(
        option(
            "--n_agents",
            help="Number of agents to simulate. " +
                 "If populate_buffer_area = y, additional agents are created to populate the buffer area."
        ).int().convert { FixedAgentNumber(it) },
        option(
            "--share_pop",
            help="Share of the population to simulate. 0.0 = 0%, 1.0 = 100%" +
                 " If populate_buffer_area = y, additional agents are created to populate the buffer area."
        ).double().convert { ShareOfPop(it) }
    ).single().default( FixedAgentNumber(1000) )
    private val n_days by option(
        help="Number of days to simulate"
    ).int().default(1)
    private val start_wd by option(
        help="First weekday to simulate. If the value is set to UNDEFINED, all simulated days will be UNDEFINED."
    ).enum<Weekday>().default(Weekday.UNDEFINED)
    private val out by option (
        help="Output file. The output format is inferred from the ending: '.json' -> Json, '.xml'-> MATSim, '.db'-> SQLite"
    ).file().default(File("output.json"))
    private val routing_mode by option(
        help = "Distance calculation method for destination choice." +
               " Either euclidean distance (BEELINE) or routed distance by car (GRAPHHOPPER)"
    ).enum<RoutingMode>().default(RoutingMode.GRAPHHOPPER)
    private val od by option(
        help="[Experimental] Path to an OD-Matrix in GeoJSON format. " +
             "The matrix is used to further calibrate the model to the area using k-factors."
    ).file(mustExist = true, mustBeReadable = true)
    private val census by option(
        help="Path to population data in GeoJSON format. " +
             "For an example of how to create such a file see python_tools/format_zensus2011.py. " +
             "Should cover the entire area, but can cover more."
    ).file(mustExist = true, mustBeReadable = true)
    private val grid_precision by option(
        help="Allowed average distance between a focus area building and its corresponding TAZ center. " +
             "The default is 150m and suitable in most cases." +
             "In the buffer area the allowed distance increases quadratically with distance. " +
             "Unit: meters"
    ).double().default(150.0)
    private val buffer by option(
        help="Distance by which the focus area (defined by GeoJSON) is buffered in order" +
             " to account for traffic generated by the surrounding. Unit: meters"
    ).double().default(0.0)
    private val seed by option(help = "RNG seed.").long()
    private val cache_dir by option(help = "Cache directory")
        .path(canBeDir = true, canBeFile = false).default(Paths.get("omod_cache/"))
    private val populate_buffer_area by option(
        help = "Determines if home locations of agents can be in the buffer area (so outside of the focus area). " +
               "If set to 'y' additional agents will be created so that the proportion of agents in and " +
               "outside the focus area is the same as in the census data. " +
               "The focus area will always be populated by n_agents agents."
    ).boolean().default(false)
    private val distance_matrix_cache_size by option(
        help = "Maximum number of entries of the distance matrix to precompute (only if routing_mode is GRAPHHOPPER). " +
               "A high value will lead to high RAM usage and long initialization times " +
               "but overall significant speed gains. The default value will use approximately 8 GB RAM at maximum."
    ).long().default(400e6.toLong())
    private val mode_choice by option(
        help = "Type of mode choice. " +
               "NONE: Returns trips with undefined modes." +
               "GTFS: Uses a logit model with public transit as an option"
    ).enum<ModeChoiceOption>().default(ModeChoiceOption.NONE)
    private val return_path_coords by option(
        help = "Whether lat/lon coordinates of chosen trip paths are returned." +
               "Paths only exist for trips with defined modes and within the focus area + buffer."
    ).boolean().default(false)
    private val population_file by option(
        help="Path to file that describes the socio-demographic makeup of the population. " +
             "Must be formatted like omod/src/main/resources/Population.json."
    ).file(mustExist = true, mustBeReadable = true)
    private val n_worker by option(
        help="Number of parallel coroutines that can be executed at the same time. " +
             "Default: Number of CPU-Cores available."
    ).int()
    private val gtfs_file by option(
        help = "Path to an General Transit Feed Specification (GTFS) for the area. " +
               "Required for public transit routing," +
               "for example if public transit is an option in mode choice. " +
               "Must be a .zip file or a directory (see https://gtfs.org/)." +
               "Recommended download platform for Germany: https://gtfs.de/"
    ).file(mustExist = true, mustBeReadable = true)
    private val calibrationParameter by CalibrationOptions().cooccurring()
    private val calibration_file by option(
        help = "Calibration input file to use for regular run." +
                "Not to be confused with --cal_out which defines the output of a calibration run."
    ).file(mustExist = true, mustBeReadable = true)

    @OptIn(ExperimentalSerializationApi::class)
    override fun run() {
        if ((census == null) && (agentNumberDefinition is ShareOfPop)) {
            throw Exception(
                "Agent population is supposed to be based on the population but no census file is provided." +
                "Consider adding a census file with --census or use --n_agents instead.")
        }
        if ((gtfs_file == null) && (mode_choice == ModeChoiceOption.GTFS)) {
            throw Exception(
                "Mode choice includes public transit as option but no GTFS file is provided." +
                "Add a gtfs file with --gtfs_file")
        }

        // Init OMOD
        val omod = Omod(
            area_geojson, osm_file,
            routingMode = routing_mode,
            odFile = od, censusFile = census,
            gridPrecision = grid_precision, bufferRadius = buffer, seed = seed,
            cache = true, cacheDir = cache_dir,
            populateBufferArea = populate_buffer_area,
            distanceCacheSize = distance_matrix_cache_size,
            populationFile = population_file,
            nWorker = n_worker,
            gtfsFile = gtfs_file
        )

        // Calibrate
        calibrationParameter?.let {
            val calibrator = TrafficCountCalibrator(
                calibrationParameter!!.cal_traffic_count_file,
                omod,
                omod.carOwnership
            )

            //calibrator.hpTune(CalibrationOption.SPSA)
            //calibrator.matrixTestRun()
            /*altPercentages = calibrator.altPercentages*/

            val calOutFile = calibrationParameter!!.cal_out
            val lossLogFile = File(
                calOutFile.parent + "/" +
                        calOutFile.name.replace(".json", "") +
                        ".losslog"
            )

            calibrator.calibrate(
                calibrationParameter!!.cal_out,
                calibrationParameter!!.cal_method,
                calibrationParameter!!.cal_activity,
                calibrationParameter!!.cal_iterations,
                lossLogFile,
                calibrationParameter!!.cal_parameter
            )
            return
        }

        if (calibration_file != null) {
            val finder = omod.destinationFinder as DestinationFinderDefault
            CalibrationInfo.read(calibration_file!!, omod.grid, omod.buildings, finder.locChoiceWeightFuns)
        }

        // Mobility demand
        val agents = when (val aND = agentNumberDefinition ) {
            is FixedAgentNumber -> omod.run(aND.value, start_wd, n_days)
            is ShareOfPop -> omod.run(aND.value, start_wd, n_days)
        }

        // Mode Choice
        omod.doModeChoice(agents, mode_choice, return_path_coords, verbose = true)

        // Store output
        logger.get()?.info("Saving results...")
        val success: Boolean
        when (out.extension) {
            "json" -> {
                FileOutputStream(out).use { f ->
                    Json.encodeToStream( agents.map { formatOutput(it) }, f)
                }
                success = true
            }
            "db" -> {
                success = writeSQLite(agents.map { formatOutput(it) }, out)
            }
            "xml" -> {
                success = writeMatSim(agents.map { formatOutput(it) }, out, n_days)
            }
            else -> {
                logger.get()?.info(
                    "Warning! output file extension ${out.extension} is not implemented." +
                    "Available output formats: .json, .db (sqlite), .xml (MATSim)" +
                    "Falling back to JSON"
                )
                val newOut = File(out.parent, out.nameWithoutExtension + ".json")
                FileOutputStream(newOut).use { f ->
                    Json.encodeToStream( agents.map { formatOutput(it) }, f)
                }
                success = true
            }
        }
        if (success) {
            logger.get()?.info("Saving results... Done!")
        }
    }
}

fun main(args: Array<String>) = Run().main(args)

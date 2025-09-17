package de.uniwuerzburg.omosim.io.osm

import crosby.binary.osmosis.OsmosisReader
import org.openstreetmap.osmosis.core.task.v0_6.RunnableSource
import org.openstreetmap.osmosis.xml.common.CompressionMethod
import org.openstreetmap.osmosis.xml.v0_6.XmlReader
import java.io.File
import java.io.FileInputStream
import java.util.Locale

object OSMSourceFactory {
    fun forPath(osmFile: File): RunnableSource {
        val name = osmFile.name.lowercase(Locale.ROOT)

        return when {
            name.endsWith(".pbf") || name.endsWith(".osm.pbf") -> {
                // PBF: reader needs an InputStream
                OsmosisReader(FileInputStream(osmFile))
            }
            name.endsWith(".osm") || name.endsWith(".osm.gz") || name.endsWith(".osm.bz2") -> {
                // XML: choose compression from file suffix
                val compression = when {
                    name.endsWith(".gz") -> CompressionMethod.GZip
                    name.endsWith(".bz2") -> CompressionMethod.BZip2
                    else -> CompressionMethod.None
                }
                // XmlReader signature: (file, enableDateParsing, compression)
                XmlReader(osmFile, /* enableDateParsing = */ false, compression)
            }
            else -> error("Unsupported OSM format: $osmFile")
        }
    }
}
package de.uniwuerzburg.omod.calibration

import de.uniwuerzburg.omod.utils.CRSTransformer
import org.locationtech.jts.geom.Coordinate
import org.locationtech.jts.geom.GeometryFactory
import org.locationtech.jts.geom.LineString
import kotlin.math.*

class Direction(
    azimuth: Double,
    centroid: Coordinate,
    transformer: CRSTransformer
) {
   private val vector: DoubleArray

   init {
       // Convert angle
       var polarAngle = 360.0 - azimuth + 90.0
       if (polarAngle >= 360.0) {
           polarAngle -= 360.0
       }

       // Create vector
       val dx = 0.01 * cos(polarAngle / 180.0 * PI)
       val dy = 0.01 * sin(polarAngle / 180.0 * PI)
       val vCoordsLatLon = GeometryFactory().createLineString(
           listOf(
               centroid,
               Coordinate(centroid.x + dy, centroid.y + dx) // Swap dx and dy because of lat-lon crs
           ).toTypedArray()
       )

       val vCoordsUTM = transformer
           .toModelCRS(vCoordsLatLon)
           .coordinates

       vector = vecFromCoords(vCoordsUTM)
   }

   private fun vecFromCoords(coords: Array<Coordinate>) : DoubleArray {
       require(coords.size == 2)

       val vector = DoubleArray(2) {0.0}
       vector[0] = coords[1].x - coords[0].x
       vector[1] = coords[1].y - coords[0].y
       return vector
   }

   private fun angleBetween(other: LineString): Double {
       val oVec = vecFromCoords(arrayOf(other.coordinates.first(), other.coordinates.last()))

       val dot = vector[0] * oVec[0] + vector[1] * oVec[1]
       val det = vector[0] * oVec[1] - vector[1] * oVec[0]
       return atan2(det, dot)
   }

   fun isSameDirection(other: LineString, leeway: Double) : Boolean {
       val leewayRad = leeway / 180.0 * PI
       val angleBe = angleBetween(other)
       return abs(angleBetween(other)) < leewayRad
   }
}
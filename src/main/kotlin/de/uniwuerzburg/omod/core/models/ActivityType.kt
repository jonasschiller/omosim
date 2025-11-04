package de.uniwuerzburg.omod.core.models

/**
 * Activity types.
 */
enum class ActivityType {
    HOME, SHARED_WORK, WORK, BUSINESS, SCHOOL, SHOPPING, OTHER;

    fun matSimName() : String {
        return when(this) {
            HOME -> "h"     // (h)ome
            SHARED_WORK ->"sw" // (s)hared (w)ork
            WORK -> "w"     // (w)ork
            BUSINESS -> "w" // (w)ork
            SCHOOL -> "e"   // (e)ducation
            SHOPPING -> "s" // (s)hopping
            OTHER -> "l"    // (l)eisure
        }
    }
}
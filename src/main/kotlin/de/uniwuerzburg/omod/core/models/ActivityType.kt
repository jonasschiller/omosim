package de.uniwuerzburg.omod.core.models

/**
 * Activity types.
 */
enum class ActivityType {
    HOME, SHARED_OFFICE, HOME_OFFICE, WORK, BUSINESS, SCHOOL, SHOPPING, OTHER;

    fun matSimName() : String {
        return when(this) {
            HOME -> "h"     // (h)ome
            HOME_OFFICE -> "h" // (h)ome
            SHARED_OFFICE ->"w" // (w)ork
            WORK -> "w"     // (w)ork
            BUSINESS -> "w" // (w)ork
            SCHOOL -> "e"   // (e)ducation
            SHOPPING -> "s" // (s)hopping
            OTHER -> "l"    // (l)eisure
        }
    }
}
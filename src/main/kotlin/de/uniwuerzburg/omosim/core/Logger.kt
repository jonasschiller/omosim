package de.uniwuerzburg.omosim.core

import org.slf4j.Logger
import org.slf4j.LoggerFactory

class PausableLogger(
    private val logger: Logger,
) {
    var on = true;

    fun get() : Logger? {
        return if (on) {
            logger
        } else {
            null
        }
    }
}

val logger: PausableLogger = PausableLogger( LoggerFactory.getLogger("de.uniwuerzburg.omosim.core") )
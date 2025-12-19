package de.uniwuerzburg.omosim.core

import de.uniwuerzburg.omosim.core.models.MobiAgent
import kotlinx.coroutines.CoroutineDispatcher
import java.util.*

/**
 * Determine the mode of trips.
 */
interface ModeChoice {
    /**
     * Determine the mode of each trip and calculate the distance and time.
     *
     * @param agents Agents with trips (usually the trips have an UNDEFINED mode at this point)
     * @param mainRng Random number generator of the main thread
     * @param dispatcher Coroutine dispatcher used for concurrency
     * @param verbose Print progressbar etc.. Doesn't affect logging.
     * @return agents. Now their trips have specified modes.
     */
    fun doModeChoice(
        agents: List<MobiAgent>, mainRng: Random, dispatcher: CoroutineDispatcher, verbose: Boolean = true
    ) : List<MobiAgent>
}
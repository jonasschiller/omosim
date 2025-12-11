package de.uniwuerzburg.omod.calibration.differentiablemodel

import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.operations.max
import kotlin.math.abs

/**
 * Generic interface that allows for the creation of functions that can build Gurobi models and
 * DifferentiableModel graphs
 *
 * T: Term
 * V: Variable
 */
interface TermBuilder<T, V> {
    fun addVar(term: T, v: V, coefficient: Double) {}
    fun addConstant(term: T, constant: Double) {}
    fun addTerm(term: T, other: T, coefficient: Double) {}
    fun new(nVars: Int) : T

    /**
     * Creates the terms that compute the following matrix multiplication:
     * L * X * R
     *
     * Where
     * L: left constant matrix
     * X: matrix filled with variable terms
     * R: right constant matrix
     *
     * @param nVars Number of variables in the problem
     * @param x matrix filled with variable terms
     * @param left L
     * @param right R
     * @param transpose if true computes: (L * X * R)^T useful for the LIVE=EVIL rule. // TODO only transposes X
     * @param relevantRCs if not null specifies which rows and columns of the result are relevant.
     * Not included columns are ignored.
     * @param cTol All terms with coefficients below this value will be ignored and not added to the result.
     */
    fun fromMatrixMult( // TODO remove default value for transpose
        nVars: Int,
        x: List<List<V>>,
        left: D2Array<Double>,
        right: D2Array<Double>,
        transpose: Boolean = true,
        relevantRCs: Set<Pair<Int, Int>>? = null,
        cTol: Double
    ) : List<List<T>> {
        val n = left.shape[0] // Assume all matrices are square and same size
        val leftMax = left.max()!! // For computation shortcut

        // Build empty result
        val result = List(n) {
            List(n) {
                this.new(nVars)
            }
        }

        // Build terms of matrix multiplication
        for (row in 0 until n) {
            for (col in 0 until  n) {
                // Ignored entry?
                if (relevantRCs != null) {
                    if (Pair(row, col) !in relevantRCs) {
                        continue
                    }
                }

                val activeEntry = result[row][col]
                for (i in 0 until n) {
                    if (right[i, col] * leftMax <= cTol) {
                        continue
                    }
                    for (j in 0 until n) {
                        val coeff = left[row, j] * right[i, col]

                        // Coefficient relevant?
                        if (abs(coeff) <= cTol) {
                            continue
                        }

                        // Add to result
                        if (transpose) {
                            this.addVar(activeEntry, x[i][j], coeff)
                        } else {
                            this.addVar(activeEntry, x[j][i], coeff)
                        }
                    }
                }
            }
        }
        return result
    }
}
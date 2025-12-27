package de.uniwuerzburg.omosim.calibration.differentiablemodel

object LinearTermBuilder: TermBuilder<LinearTerm, Term> {
    override fun addVar(term: LinearTerm, v: Term, coefficient: Double) {
        term.addTerm(v, coefficient)
    }

    override fun addConstant(term: LinearTerm, constant: Double) {
        term.addConstant(constant)
    }

    override fun addTerm(term: LinearTerm, other: LinearTerm, coefficient: Double) {
        term.addTerm(other, coefficient)
    }

    override fun new(nVars: Int): LinearTerm {
        return LinearTerm(nVars)
    }
}
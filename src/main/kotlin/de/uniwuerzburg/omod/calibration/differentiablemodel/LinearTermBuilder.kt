package de.uniwuerzburg.omod.calibration.differentiablemodel

object LinearTermBuilder: TermBuilder<LinearTerm, Term> {
    override fun addTerm(term: LinearTerm, v: Term, coefficient: Double) {
        term.addTerm(v, coefficient)
    }

    override fun addConstant(term: LinearTerm, constant: Double) {
        term.addConstant(constant)
    }

    override fun addSum(term: LinearTerm, sum: LinearTerm, coefficient: Double) {
        term.addTerm(sum, coefficient)
    }

    override fun createSum(nVars: Int): LinearTerm {
        return LinearTerm(nVars)
    }

    override fun addTermToSum(s: LinearTerm, v: Term, coefficient: Double) {
        s.addTerm(v, coefficient)
    }

    override fun addConstToSum(s: LinearTerm, const: Double) {
        s.addConstant(const)
    }
}
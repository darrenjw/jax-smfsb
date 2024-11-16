#!/usr/bin/env python3
# smfsbSbml.py
# Import SBML models into an smfsb Spn

import libsbml
import sys
from jsmfsb import Spn
import jax.numpy as jnp
from sbmlsh import mod2sbml


def mod_to_spn(filename, verb=False):
    """Convert an SBML-shorthand model into a Spn object

    Read a file containing a model in SBML-shorthand and convert into
    an Spn object for simulation and analysis.

    Parameters
    ----------
    filename: string
        String name of file containing the model
    verb: boolean
        Output some debugging info

    Returns
    -------
    An Spn object

    Examples
    --------
    >>> import smfsb
    >>> myMod = smfsb.mod2spn("myModel.mod")
    >>> step = myMod.step_gillespie()
    """
    try:
        s = open(filename, "r")
    except:
        sys.stderr.write("Error: failed to open " + filename + "\n")
        sys.exit(1)
    p = mod2sbml.Parser()
    d = p.parseStream(s)
    m = d.getModel()
    if m is None:
        sys.stderr.write("Error: can't extract SBML model\n")
        sys.exit(1)
    return model_to_spn(m, verb)


def shorthand_to_spn(sh_string, verb=False):
    """Convert an SBML-shorthand model string into a Spn object

    Parse a string containing a model in SBML-shorthand and convert into
    an Spn object for simulation and analysis.

    Parameters
    ----------
    sh_string: string
        String containing the model
    verb: boolean
        Output some debugging info

    Returns
    -------
    An Spn object

    Examples
    --------
    >>> import smfsb
    >>> file = open('myModel.mod', 'r')
    >>> myModStr = file.read()
    >>> file.close()
    >>> myMod = smfsb.sh2spn(myModStr)
    >>> step = myMod.step_gillespie()
    """
    p = mod2sbml.Parser()
    d = p.parse(sh_string)
    m = d.getModel()
    if m is None:
        sys.stderr.write("Error: couldn't parse the shorthand string\n")
        sys.exit(1)
    return model_to_spn(m, verb)


def file_to_spn(filename, verb=False):
    """Convert an SBML model into a Spn object

    Read a file containing a model in SBML and convert into
    an Spn object for simulation and analysis.

    Parameters
    ----------
    filename: string
        String name of file containing the model
    verb: boolean
        Output some debugging info

    Returns
    -------
    An Spn object

    Examples
    --------
    >>> import smfsb
    >>> myMod = smfsb.mod2spn("myModel.xml")
    >>> step = myMod.step_gillespie()
    """
    d = libsbml.readSBML(filename)
    m = d.getModel()
    if m is None:
        sys.stderr.write("Can't parse SBML file: " + filename + "\n")
        sys.exit(1)
    return model_to_spn(m, verb)


def model_to_spn(m, verb=False):
    """Convert a libSBML model into a Spn object

    Convert a libSBML model into a Spn object for simulation and analysis.

    Parameters
    ----------
    m: model
        A libsbml model (not document) object
    verb: boolean
        Output some debugging info

    Returns
    -------
    An Spn object

    Examples
    --------
    >>> import smfsb
    >>> import libsbml
    >>> d = libsbml.readSBML("myModel.xml")
    >>> m = d.getModel()
    >>> myMod = smfsb.model_to_spn(m)
    >>> step = myMod.step_gillespie()
    """
    # Species and initial amounts
    ns = m.getNumSpecies()
    if verb:
        print(str(ns) + " species")
    ml = []
    nl = []
    for i in range(ns):
        s = m.getSpecies(i)
        nl += [s.getId()]
        ml += [s.getInitialAmount()]
    if verb:
        print(nl)
        print(ml)
    # Compartments
    nc = m.getNumCompartments()
    cd = {}
    for i in range(nc):
        comp = m.getCompartment(i)
        cd[comp.getId()] = comp.getVolume()
    if verb:
        print(cd)
    # Global parameters
    ngp = m.getNumParameters()
    gpd = {}
    for i in range(ngp):
        param = m.getParameter(i)
        gpd[param.getId()] = param.getValue()
    if verb:
        print(gpd)
    # Reactions
    nr = m.getNumReactions()
    if verb:
        print(str(nr) + " reactions")
    pre = jnp.zeros((nr, ns))
    post = jnp.zeros((nr, ns))
    rn = []
    kl = []
    lpl = []
    for i in range(nr):
        r = m.getReaction(i)
        rn += [r.getId()]
        n_pre = r.getNumReactants()
        for j in range(n_pre):
            sr = r.getReactant(j)
            sto = sr.getStoichiometry()
            pre = pre.at[i, nl.index(sr.getSpecies())].set(sto)
        n_post = r.getNumProducts()
        for j in range(n_post):
            sr = r.getProduct(j)
            sto = sr.getStoichiometry()
            post = post.at[i, nl.index(sr.getSpecies())].set(sto)
        kli = r.getKineticLaw()
        kl += [libsbml.formulaToString(kli.getMath())]
        nlp = kli.getNumLocalParameters()
        lpd = {}
        for j in range(nlp):
            param = kli.getLocalParameter(j)
            lpd[param.getId()] = param.getValue()
        lpl += [lpd]
    if verb:
        print(rn)
        print("Pre:")
        print(pre)
        print("Post:")
        print(post)
        print(kl)
        print(lpl)
    gpd.update(cd)

    def haz(x, t):
        h = jnp.zeros(nr)
        xd = dict(zip(nl, x))
        glob = gpd.copy()
        glob.update(xd)
        for i in range(nr):
            h = h.at[i].set(eval(kl[i], glob, lpl[i]))
        return h

    spn = Spn(nl, rn, pre, post, haz, ml)
    spn.comp = cd
    spn.gp = gpd
    spn.kl = kl
    spn.lp = lpl
    return spn


# eof

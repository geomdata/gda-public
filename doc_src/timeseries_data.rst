Time-Series Data
================

If your data is parametrized by a single _time_ variable, then it is essentially
one-dimensional.  Data of this type are analyzed in the `timeseries` module.

There are two distinct but related cases:

 * The data is a plain function, :math:`\mathrbb{R} \to \mathbb{R}`. These are studied with the :class:`timeseries.Signal` class.
 * The data is an embedded curve, :math:`\mathrbb{R} \to \mathbb{R}^d`. These are studied with the :class:`timeseries.SpaceCurve` class.

Of course, each component of a :class:`timeseries.SpaceCurve` could be analyzed
as :class:`timeserie.Signal`, and a :class:`timeseries.Signal` can be thought
of as a :class:`timeseries.SpaceCurve` with :math:`d=1`.  However, the
geometric considerations are often quite different between a simple
amplitude-versus-time signal, and a trajectory in space.


 .. autosummary::
     :toctree: generated/
     
     timeseries
     timeseries.Signal
     timeseries.SpaceCurve



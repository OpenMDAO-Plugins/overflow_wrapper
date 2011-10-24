

This is the OpenMDAO component wrapper for OVERFLOW (OVERset grid FLOW solver).

It is a simple file-wrap, where the namelists are available to OpenMDAO as
component inputs. During execution, the overflow input file is generated
and then overflow is executed. At present, no information is parsed from the 
solution file.

Note: this installation does not include OVERFLOW, which is a proprietary research
code. You must obtain that separately through the proper channels:

http://aaac.larc.nasa.gov/~buning/codes.html


To view the Sphinx documentation for this distribution, type:

plugin_docs overflow_wrapper


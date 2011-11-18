"""
OpenMDAO wrapper for the Overflow CFD solver
created based on Overflow 2.2c
"""

# pylint: disable-msg=E0611,F0401,E1101
from numpy import int64 as numpy_int64
from numpy import float64 as numpy_float64
from numpy import array, zeros

from openmdao.main.api import VariableTree, FileMetadata
from openmdao.lib.components.api import ExternalCode
from openmdao.lib.datatypes.api import Array, Bool, Enum, Float, Int, List, \
                                       Slot, Str
from openmdao.lib.datatypes.domain import Vector, read_plot3d_grid
from openmdao.util.log import NullLogger
from openmdao.util.namelist_util import Namelist
from openmdao.util.stream import Stream


class Namelist_GLOBAL(VariableTree):
    """VariableTree for the GLOBAL namelist"""
    
    nsteps = Int(0, desc='Number of (fine-grid) steps to advance solution. Use zero for input check.')
    restrt = Bool(False, desc='True - Read restart flowfield from file q.restart;\n'
                              'False - Start from initial free-stream flowfield.')
    nsave = Int(100, desc='>=0 - Save the overall solution to file q.save every how many steps.\n'
                          '<0 - Save the solution to file q.step# every how many steps.\n'
                          'Note that files are saved as q.step# for dynamic or adaption cases regardless of the sign of NSAVE.')
    save_hiorder = Enum(2, [-1, 0, 1, 2], 
                        desc='Controls whether Q**(n-1) data for 2nd-order restarts is written to q.saveand q.step# files:\n'
                                '-1 - Always include Q**(n-1).\n'
                                '0 - Never include Q**(n-1).\n'
                                '1 - Only include Q**(n-1) for final q.save.\n'
                                '2 - Always include Q**(n-1) for q.save, never for q.step# if NSAVE<0.')
    istart_qavg = Enum(0, [0, 1], 
                       desc='0 - Do not save Q average/perturbation data.\n'
                            '>0 - Start saving Q average and (rho,u,v,w,p) perturbation data at step ISTART_QAVG. Write to file q.avg whenever q.save is written. Note that average/perturbation data starts fresh every run.')
    nfomo = Int(10, desc='Compute aerodynamic forces and moments every how many steps.')
    nqt = Enum(0, [0, 100, 101, 102, 202, 203, 204, 205], 
               desc='Global turbulence model type declaration:\n'
                       '0 - Algebraic or no turbulence model.\n'
                       '100 - Baldwin-Barth (1-eq) model.\n'
                       '101 - Spalart-Allmaras (1-eq) model with trip line specification.\n'
                       '102 - Spalart-Allmaras (1-eq) model.\n'
                       '202 - k-omega (2-eq) model (DDADI left-hand side).\n'
                       '203 - SST (2-eq) model (DDADI left-hand side).\n'
                       '204 - k-omega (2-eq) model (SSOR left-hand side).\n'
                       '205 - SST (2-eq) model (SSOR left-hand side).')
    nqc = Int(0,desc='Variable gamma model type declaration (number of species):\n'
                      '0 - Constant gamma, 1-gas variable gamma, or 2-gas variable gamma with mixing based on stagnation enthalpy.\n'
                      '>=2 - Multiple gas variable gamma based on solution of NQC species continuity equations.')
    multig = Bool(False, desc='Flag to enable/disable multigrid acceleration.')
    fmg = Bool(False, desc='Flag to enable/disable grid sequencing.')
    # TODO - Something sets the size of this array, and defaults it to 300 for all course levels.
    fmgcyc = Array(dtype=numpy_int64,
                   desc='Number of steps to take on coarser grid levels during grid sequencing. Here index 1 is the coarsest level, 2 the next finer, etc.')
    nglvl = Int(3, desc='Number of multigrid and/or grid sequencing levels to use.')
    # DONE - Manual says no default value, Presumably this means zero?
    tphys = Float(-9999.0, 
                  desc='Starting physical time (overrides value from q.restart). Set to -9999 to use default in q.restart.')
    dtphys = Float(0, desc='Physical time-step (based on Vref).')
    nitnwt = Int(0, desc='Number of Newton/dual subiterations per physical time-step, or 0 for no subiteration.')
    fsonwt = Float(2.0, low=1.0, high=2.0,
                   desc='1.0 - First-order time-advance for Newton/dual subiteration.\n'
                        '2.0 - Second-order time-advance for Newton/dual subiteration.\n'
                        'Intermediate values allowed.')
    # Not used
    #ordnwt = Int(0, desc='Not used. Number of orders of convergence per time-step for Newton/dual subiteration.')
    rf = Float(0.0, units='rad/s',
               desc='Global coordinate system z-rotation speed (rad/time, based on Vref)')
    cdisc = Bool(False, 
                 desc='True - Expect to read a NAMELIST input file overdisc.con, containing CDISC inverse design control information. This file will be updated by OVERFLOW.\n'
                      'False - Do not read or write any CDISC information.')
    grdwts = Bool(False, 
                  desc='True - Use grid timing information in grdwghts.restart for MPI load balancing, if available. (Equivalent to USEFLE in $GROUPS.)\n'
                       'False - Use normal load-balancing algorithm.')
    max_grid_size = Int(0, 
                        desc='0 - Use automatic grid splitting algorithm for load balancing.\n'
                             '>0 - Specified (weighted) size limit for split grids.\n'
                             '<0 - Do not split grids.\n'
                             '(Sets default MAXNB and MAXGRD in $GROUPS.)')
    nobomb = Bool(False, desc='Inhibit writing q.bombfile if solution procedure fails.')
    conserve_mem = Bool(False, 
                        desc='Conserve memory by recomputing metrics and regenerating coarse-level grids every iteration.')
    debug = Enum(0, [0, 1, 2, 3], desc='0 - Normal run.\n'
                         '1 - Write turbulence model debug file q.turb and quit.\n'
                         '2 - Write timestep debug file q.time and quit.\n'
                         '3 - Write residual debug file q.resid and quit.')


class Namelist_OMIGLB(VariableTree):
    """VariableTree for the OMIGB namelist. Contains global inputs for 
    OVERFLOW-D. (OVERFLOW-D only.)
    """
    
    irun = Enum(0, [0, 1, 2], 
                desc='0 - Do a complete run.\n'
                        '1 - Run only through off-body (brick) grid generation.\n'
                        '2 - Run only through overset grid connectivity (DCF).')
    i6dof = Enum(0, [0, 1, 2], desc='0 - Body motion is defined by user-defined USER6 routine.\n'
                         '1 - Body motion is defined by inputs in $SIXINP.\n'
                         '2 - Body motion is defined by GMP interface (files Config.xml and Scenario.xml).')
    dynmcs = Bool(False, desc='Enable/disable body motion.')
    nadapt = Int(0, desc='0 - Do not regenerate off-body grids.\n'
                         '>0 - Regenerate off-body grids every NADAPT steps, based on geometry proximity and solution error estimation.\n'
                         '<0 - Regenerate off-body grids every NADAPT steps, based on geometry proximity only.')
    sigerr = Float(2.0, desc='Solution error order for adaption.')
    r_coef = Float(1.0, desc='Coefficient of restitution for collisions.')
    lfringe = Int(0, desc='|LFRINGE| is the number of fringe points for near-body grids and hole boundaries. If LFRINGE<0, do not revert double- and higher-fringe orphan points to field points. [Determined from numerical scheme (all grids).]')
    ibxmin = Int(47, desc='Boundary condition type for Xmin far-field boundaries.')
    ibxmax = Int(47, desc='Boundary condition type for Xmax far-field boundaries.')
    ibymin = Int(47, desc='Boundary condition type for Ymin far-field boundaries.')
    ibymax = Int(47, desc='Boundary condition type for Ymax far-field boundaries.')
    ibzmin = Int(47, desc='Boundary condition type for Zmin far-field boundaries.')
    ibzmax = Int(47, desc='Boundary condition type for Zmax far-field boundaries.')
    laminar_ob = Bool(False, 
                      desc='Force laminar flow in off-body grids. (Applicable to NQT=100, but not NQT=101.)')


class Namelist_GBRICK(VariableTree):
    """VariableTree for the GBRICK namelist. Contains off-body grid generation
    inputs. (OVERFLOW-D only.)
    """
    
    obgrids = Bool(True, desc='Allow or inhibit off-body grids.')
    # TODO - This defaults to IGSIZE/2. We need to not print this to the file if it's not set.
    max_brick_size = Int(5000000, 
                         desc='>0 - Maximum off-body grid size.\n'
                              '<=0 - No limit on off-body grid size.\n'
                              'Defaults to IGSIZE/2')
    ds = Float(0.0, desc='Spacing of level-1 (finest) off-body grids. THIS MUST BE SPECIFIED.')
    dfar = Float(5.0, desc='Distance to far-field boundaries.')
    # TODO - These next 3 default to center of near-body grids. Need to not print this if not set.
    xncen = Float(0.0, desc='Center of off-body grid system. Must be specified for repeatable off-body grid generation with body motion.')
    yncen = Float(0.0, desc='Center of off-body grid system. Must be specified for repeatable off-body grid generation with body motion.')
    zncen = Float(0.0, desc='Center of off-body grid system. Must be specified for repeatable off-body grid generation with body motion.')
    #chrlen = Float(1.0, desc='Characteristic body length for off-body grid generation. [Currently not used.]')
    # TODO - For this next batch, seems you can only set min or max.
    i_xmin = Enum(0, [0, 1], 
                     desc='0 - Xmin far-field boundary will be determined by DFAR.\n'
                          '1 - Xmin boundary will be specified by P_XMIN, resp.\n'
                          '(Only one may be specified in the x-direction.)')
    i_xmax = Enum(0, [0, 1], 
                     desc='0 - Xmax far-field boundary will be determined by DFAR.\n'
                          '1 - Xmax boundary will be specified by P_XMAX, resp.\n'
                          '(Only one may be specified in the x-direction.)')
    i_ymin = Enum(0, [0, 1], 
                     desc='0 - Ymin far-field boundary will be determined by DFAR.\n'
                          '1 - Ymin boundary will be specified by P_YMIN, resp.\n'
                          '(Only one may be specified in the y-direction.)')
    i_ymax = Enum(0, [0, 1], 
                     desc='0 - Ymax far-field boundary will be determined by DFAR.\n'
                          '1 - Ymax boundary will be specified by P_YMAX, resp.\n'
                          '(Only one may be specified in the y-direction.)')
    i_zmin = Enum(0, [0, 1], 
                     desc='0 - Zmin far-field boundary will be determined by DFAR.\n'
                          '1 - Zmin boundary will be specified by P_ZMIN, resp.\n'
                          '(Only one may be specified in the z-direction.)')
    i_zmax = Enum(0, [0, 1], 
                     desc='0 - Zmax far-field boundary will be determined by DFAR.\n'
                          '1 - Zmax boundary will be specified by P_ZMAX, resp.\n'
                          '(Only one may be specified in the z-direction.)')
    p_xmin = Float(0.0, desc='Physical location for Xmin, Xmax off-body grid boundary, if corresponding I_XMIN, I_XMAX != 0.')
    p_xmax = Float(0.0, desc='Physical location for Xmin, Xmax off-body grid boundary, if corresponding I_XMIN, I_XMAX != 0.')
    p_ymin = Float(0.0, desc='Physical location for Ymin, Ymax off-body grid boundary, if corresponding I_YMIN, I_YMAX != 0.')
    p_ymax = Float(0.0, desc='Physical location for Ymin, Ymax off-body grid boundary, if corresponding I_YMIN, I_YMAX != 0.')
    p_zmin = Float(0.0, desc='Physical location for Zmin, Zmax off-body grid boundary, if corresponding I_ZMIN, I_ZMAX != 0.')
    p_zmax = Float(0.0, desc='Physical location for Zmin, Zmax off-body grid boundary, if corresponding I_ZMIN, I_ZMAX != 0.')
    minbuf = Int(4, desc='Minimum buffer width of points at each level.')
    # TODO - This is determined from off-body numerical shceme, or from brkset.restart file'
    ofringe = Int(0, desc='Number of fringe points for off-body grids. [Determined from off-body numerical scheme or from brkset.restart file.]')


class Namelist_BRKINP(VariableTree):
    """VariableTree for the BRKINP namelist. Contains user-specified proximity
    regions. (OVERFLOW-D only.)
    """

    nbrick = Int(0, desc='Number of user-specified proximity regions. If NBRICK<0, user must specify ALL proximity regions (i.e., geometry will not be used).')
    xbrkmin = Array(dtype=numpy_float64,
                    desc='X-range of user-specified proximity region(s).')
    xbrkmax = Array(dtype=numpy_float64,
                    desc='X-range of user-specified proximity region(s).')
    ybrkmin = Array(dtype=numpy_float64,
                    desc='Y-range of user-specified proximity region(s).')
    ybrkmax = Array(dtype=numpy_float64,
                    desc='Y-range of user-specified proximity region(s).')
    zbrkmin = Array(dtype=numpy_float64,
                    desc='Z-range of user-specified proximity region(s).')
    zbrkmax = Array(dtype=numpy_float64,
                    desc='Z-range of user-specified proximity region(s).')
    # TODO - This array defaults to ones(nbrick) if not specified
    ibdytag = Array(dtype=numpy_int64,
                    desc='>0 - Proximity region will be linked to this body ID for dynamic motion.\n'
                         '0 - Proximity region will have no body transformations.')
    # TODO - This array defaults to zeros(nbrick) if not specified
    deltas = Array(dtype=numpy_float64,
                   desc='Distance to expand proximity region (in all directions).')


class Namelist_GROUPS(VariableTree):
    """VariableTree for the GROUPS namelist. Contains load balance input
    (OVERFLOW-D only.)
    """

    usefle = Bool(False, 
                  desc='True - Use grid timing information in grdwghts.restart for MPI load balancing, if available. (Equivalent to GRDWTS in $GLOBAL.)\n'
                       'False - Use normal load-balancing algorithm.')
    # TODO - This can be set by max_grid_size? Need to find out more about this.
    maxnb = Int(0, 
                desc='0 - Use automatic splitting algorithm for near-body grid load balancing.\n'
                     '>0 - Specified (weighted) size limit for split grids.\n'
                     '<0 - Do not split grids.\n'
                     '(Can be set by MAX_GRID_SIZE in $GLOBAL.)')
    # TODO - This can be set by max_grid_size? Need to find out more about this.
    maxgrd = Int(0, 
                 desc='0 - Use automatic splitting algorithm for off-body grid load balancing.\n'
                      '>0 - Specified (weighted) size limit for split grids.\n'
                      '<0 - Do not split grids.\n'
                      '(Can be set by MAX_GRID_SIZE in $GLOBAL.)')
    wghtnb = Float(1.0, 
                   desc='Weight-factor for near-body grids vs. off-body grids in normal load-balancing algorithm.')
    igsize = Int(10000000, 
                 desc='Maximum group size during off-body grid adaption.')

    
class Namelist_DCFGLB(VariableTree):
    """VariableTree for the DCFGLB namelist. Contains DCF input
    (OVERFLOW-D only.)
    """

    dqual = Float(1.0, desc='Acceptable "quality" of donor interpolation stencils.')
    morfan = Enum(1, [1, 0], desc='1/0 - Enable/disable viscous stencil repair.')
    norfan = Int(5, desc='Number of points above viscous wall subject to viscous stencil repair.')
    
    
# TODO - Multiple occurances of this namelist, so use functional interface to add.
class Namelist_XRINFO(VariableTree):
    """VariableTree for the XRINFO namelist. (X-ray input, repeat per X-ray cutter)
    (OVERFLOW-D only.)
    """

    idxray = Int(0, 
                 desc='X-ray to be used for this cutter. (Note that X-rays may be used in multiple cutters.)\n'
                 'THIS INPUT MUST BE SPECIFIED')
    igxlist = List([], Int(), 
                   desc='Specify a list of grids to be cut by this cutter (a grid number of -1 refers to all off-body grids).')
    # TODO - Default should really be none, so will have tocheck if igxlist is empty.
    igxbeg = Int(0, desc='Or specify beginning and ending grids to be cut by this cutter.')
    igxend = Int(0, desc='Or specify beginning and ending grids to be cut by this cutter.')
    xdelta = Float(0.0, desc='Hole will extend XDELTA from the X-rayed surface.')


class Namelist_FLOINP(VariableTree):
    """VariableTree for the FLOINP namelist. Contains Flow parameters.
    """

    fsmach = Float(0.0, desc='Freestream Mach number (M_inf).')
    # DONE - Default for this is fsmach.
    refmach = Float(0.0, desc='Reference Mach number (M_ref). Defaults to fsmach if this is set to zero.')
    alpha = Float(0.0, units='deg', desc='Angle-of-attack (alpha), deg.')
    beta = Float(0.0, units='deg', desc='Sideslip angle (beta), deg.')
    rey = Float(0.0, desc='Reynolds number (Re) (based on V_ref and grid length unit).')
    tinf = Float(518.7, units='degR',
                 desc='Freestream static temperature (T_inf), deg. Rankine.')
    gaminf = Float(1.4, desc='Freestream ratio of specific heats (gamma_inf).')
    pr = Float(0.72, desc='Prandtl number (Pr).')
    prt = Float(0.9, desc='Turbulent Prandtl number (Prt).')
    retinf = Float(0.1, desc='Freestream turbulence level (mu_t/mu_l)_inf for 1-or 2-eq turbulence models.')
    xkinf = Float(1.0e-6, desc='Freestream turbulent kinetic energy (k_inf/V_ref**2) for 2-eq turbulence models.')
    targcl = Bool(False, desc='Enable the target C_L-driver option.')
    cltarg = Float(0.0, desc='Value of C_L the code will try to match.')
    clalph = Float(0.1, desc='Fixed value of d_CL/d_alpha used to update ALPHA.')
    ntarg = Int(10, desc='Number of steps between ALPHA corrections, with the following exceptions: corrections are not done during grid sequencing, and corrections are not done on the first or last fine-grid steps.')
    ctp = Float(0.0, desc='Rotor thrust coefficient (for BC type 37).')
    aspctr = Float(1.0, desc='Rotor radius (for BC type 37).')
    froude = Float(0.0, desc='Froude number (gravity term) (Fr) (based on Vrefand grid length unit).')
    gvec = Array(array([0.0, 0.0, 1.0]), dtype=numpy_float64,
                 desc='Unit up-vector for FROUDE gravity term (Note that this vector is taken verbatim. It is not modified internally by the angle-of-attack, since other orientation angles (such as bank angle) are not known.')

    
class Namelist_VARGAM(VariableTree):
    """VariableTree for the VARGAM namelist. Contains variable gamma input.
    """

    igam = Int(0, 
                  desc='Options for specifying calculation of gamma when notsolving species continuity equations (i.e., NQC<2):\n'
                       '0 - Use a constant gamma value of GAMINF.\n'
                       '1 - Single gas with temperature variation of gamma computed using ALT0-4, AUT0-4.\n'
                       '2 - Two gases with temperature variation of gamma computed using ALT0-4, AUT0-4; all gas 1 below HT1, all gas 2 above HT2, linear mix in between.')
    ht1 = Float(10.0, 
                desc='Total enthalpy ratio h_0/h_0inf below which the mixture is all gas 1.')
    ht2 = Float(10.0, 
                desc='Total enthalpy ratio h_0/h_0inf below which the mixture is all gas 2.')
    # TODO - This stuff hinges on setting for igam. May need to resize dynamically.
    # TODO - default is 1 for gas 1, 0 for all others.
    scinf = Array(array([1]), dtype=numpy_int64,
                  desc='Freestream species mass fraction c_i_inf.')
    smw = Array(array([1.0]), dtype=numpy_float64,
                desc='Species molecular weight MWi, or normalized molecular weight MWi/MW_inf (if preferred).')
    # TODO - Default value comes from floinp.gaminf. What if user wants this val to differ?
    alt0 = Array(array([1.0]), dtype=numpy_float64, units='degR', 
                               low=540, high=1800, exclude_low=True, exclude_high=True,
                               desc='Lower temperature range polynomial coefficient a0 (540 degR < T < 1800 degR).')
    alt1 = Array(array([0.0]), dtype=numpy_float64, units='degR',
                               low=540, high=1800, exclude_low=True, exclude_high=True,
                               desc='Lower temperature range polynomial coefficient a1 (540 degR < T < 1800 degR).')
    alt2 = Array(array([0.0]), dtype=numpy_float64, units='degR',
                               low=540, high=1800, exclude_low=True, exclude_high=True,
                               desc='Lower temperature range polynomial coefficient a2 (540 degR < T < 1800 degR).')
    alt3 = Array(array([0.0]), dtype=numpy_float64, units='degR',
                               low=540, high=1800, exclude_low=True, exclude_high=True,
                               desc='Lower temperature range polynomial coefficient a3 (540 degR < T < 1800 degR).')
    alt4 = Array(array([0.0]), dtype=numpy_float64, units='degR',
                               low=540, high=1800, exclude_low=True, exclude_high=True,
                               desc='Lower temperature range polynomial coefficient a4 (540 degR < T < 1800 degR).')
    # TODO - Default value for this is whatever is in alt0.
    aut0 = Array(array([1.0]), dtype=numpy_float64, units='degR', 
                               low=1800, high=9000, exclude_low=True, exclude_high=True,
                               desc='Upper temperature range polynomial coefficient a0 (1800 degR < T < 9000 degR).')
    aut1 = Array(array([0.0]), dtype=numpy_float64, units='degR', 
                               low=1800, high=9000, exclude_low=True, exclude_high=True,
                               desc='Upper temperature range polynomial coefficient a1 (1800 degR < T < 9000 degR).')
    aut2 = Array(array([0.0]), dtype=numpy_float64, units='degR', 
                               low=1800, high=9000, exclude_low=True, exclude_high=True,
                               desc='Upper temperature range polynomial coefficient a2 (1800 degR < T < 9000 degR).')
    aut3 = Array(array([0.0]), dtype=numpy_float64, units='degR', 
                               low=1800, high=9000, exclude_low=True, exclude_high=True,
                               desc='Upper temperature range polynomial coefficient a3 (1800 degR < T < 9000 degR).')
    aut4 = Array(array([0.0]), dtype=numpy_float64, units='degR', 
                               low=1800, high=9000, exclude_low=True, exclude_high=True,
                               desc='Upper temperature range polynomial coefficient a4 (1800 degR < T < 9000 degR).')
    sigl = Array(array([1.0]), dtype=numpy_float64, 
                               desc='Laminar diffusion coefficient sigma_l.')
    sigt = Array(array([1.0]), dtype=numpy_float64, 
                               desc='Turbulent diffusion coefficient sigma_t.')

    
# The following namelists are added in Grid

class Namelist_NITERS(VariableTree):
    """VariableTree for the NITERS namelist. Subiterations per grid.
    Contained in the Grid VariableTree.
    """

    iter = Int(1, 
               desc='Number of flow solver iterations per step. (Each flow solver iteration performs ITERT turbulence model iterations and ITERC species continuity iterations.)')


class Namelist_METPRM(VariableTree):
    """VariableTree for the METPRM namelist. Numerical methods solution.
    Contained in the Grid VariableTree.
    """

    irhs = Enum(0, [0, 2, 3, 4, 5], 
                  desc='0 - Central difference Euler terms.\n'
                       '2 - Yee symmetric TVD scheme.\n'
                       '3 - Liou AUSM+ flux split scheme.\n'
                       '4 - Roe upwind scheme.\n'
                       '5 - HLLC upwind scheme.')
    ilhs = Enum(2, [0, 1, 2, 3, 4, 5, 6], 
                   desc='0 - ARC3D Beam-Warming block tridiagonal scheme.\n'
                        '1 - F3D Steger-Warming 2-factor scheme.\n'
                        '2 - ARC3D diagonalized Beam-Warming scalar pentadiagonal scheme.\n'
                        '3 - LU-SGS algorithm.\n'
                        '4 - D3ADI algorithm with Huang subiteration.\n'
                        '5 - ARC3D Beam-Warming with Steger-Warming flux split jacobians.\n'
                        '6 - SSOR algorithm (with subiteration).')
    # DONE - Funky default settings here: [10 for ILHS=6, 3 for ILHS=4]
    ilhsit = Int(-1, 
                 desc='Number of subiterations for D3ADI or SSOR. Set to -1 to let Overflow determine the default from [10 for ILHS=6, 3 for ILHS=4]')
    idiss = Enum(3, [2, 3, 4], 
                    desc='2 - ARC3D dissipation scheme (2nd-, 4th-order dissipation on RHS and LHS).\n'
                         '3 - TLNS3D dissipation scheme (same as IDISS=2, but smooth rho*h_0 instead of rho*e_0).\n'
                         '4 - Matrix dissipation scheme (see additional parameters VEPSL, VEPSN, ROEAVG in $SMOINP).')
    ilimit = Enum(1, [1, 2, 3, 4], 
                     desc='Limiter for upwind Euler terms (IRHS=3-5):\n'
                          '1 - Koren limiter.\n'
                          '2 - Minmod limiter.\n'
                          '3 - van Albada limiter.\n'
                          '4 - WENO5M scheme (FSO>3 only).\n'
                          'See DELTA for further control.')
    bimin = Float(1.0, 
                  desc='1.0 - Disable low-Mach preconditioning.\n'
                       '-1.0 - Enable low-Mach preconditioning; reset BIMIN to 3xMref2.\n'
                       '<1.0 - Enable low-Mach preconditioning with specified BIMIN.')
    ssor_relax = Float(0.9, 
                       desc='Relaxation factor for SSOR schemes (flow eqns, turb models, species eqns).')
    q_limit = Bool(True, 
                   desc='True - Limit Q update procedure to try to keep density and energy from going negative.\n'
                        'False - Use simple Q update procedure (from OVERFLOW 1.8).')
    # DONE - This default is linked to the one in Global.
    multig = Bool(False, desc='Flag to enable/disable multigrid acceleration.')
    smoop = Float(0.0, desc='Smoothing coefficient for prolongation of coarse-grid solution onto next-finer level during grid sequencing.')
    smooc = Float(0.0, desc='Smoothing coefficient for multigrid correction before interpolation onto next-finer level.')
    smoor = Float(0.0, desc='Smoothing coefficient for multigrid residual before restricting to the next-coarser level.')
    corsvi = Bool(True, desc='Enable/disable computation of viscous terms on coarse grid levels.')
    recmut = Bool(False, desc='Recompute mu_t on finest level during multigrid.')


class Namelist_TIMACU(VariableTree):
    """VariableTree for the TIMACU namelist. Time accuracy.
    Contained in the Grid VariableTree.
    """

    itime = Enum(1, [0, 1, 2, 3, 4], 
                    desc='Time-step scaling flag:\n'
                         '0 - Constant time-step, no scaling (used for simple time-stepping or Newton subiteration).\n'
                         '1 - Local time-step scaling (with 0.005 dimensional fudge-factor).\n'
                         '2 - Local time-step scaling (with no fudge-factor).\n'
                         '3 - Constant CFL number (based on CFLMAX value). This uses the sum of the (max) eigenvalue in each coordinate direction to determine the local CFL number. All other uses of CFLMIN/CFLMAX use the maximum eigenvalue to determine the CFL number.\n'
                         '4 - Same as ITIME=3, but adjust timestep scaling based on local cell Reynolds number.')
    dt = Float(0.5, desc='Time-step factor.')
    cflmin = Float(0.0, desc='Minimum CFL number.')
    cflmax = Float(0.0, desc='Maximum CFL number.')
    tfoso = Enum(1.0, [1.0, 2.0, 0.5, 1.9], 
                      desc='Order of time-accuracy, when using simple time-stepping (NITNWT=0):\n'
                           '1.0 - 1st-order time-accuracy (Euler implicit scheme).\n'
                           '2.0 - 2nd-order time-accuracy (trapezoidal scheme).\n'
                           'Other values allowed; 0.5, 1.9 are OK.')


class Namelist_SMOACU(VariableTree):
    """VariableTree for the SMOACU namelist. Smoothing parameters.
    Contained in the Grid VariableTree.
    """

    ispec = Enum(2, [-1, 1, 2, 3], 
                    desc='Dissipation scaling flag; single value to specify ISPECJ,ISPECK,ISPECL:\n'
                         '-1 - Sum spectral radii in J, K, and L.\n'
                         '1 - Constant coefficient dissipation.\n'
                         '2 - Spectral radius in J, K, or L.\n'
                         '3 - Weighted average of J, K, L spectral radii (TLNS3D-type).')
    # TODO - Funky initial conditions
    smoo = Float(1.0, low=0.0, high=1.0,
                      desc='0.0 - Spectral radius is computed normally, as |U|+kc.\n'
                           '1.0 - Sound speed c is replaced by ||V||/M_ref, reducing smoothing in low-speed regions.\n'
                           'Intermediate values are allowed.')
    dis2 = Float(2.0, desc='2nd-order smoothing coefficient.')
    dis4 = Float(0.04, desc='4th-order smoothing coefficient.')
    # DONE - Funky initial conditions
    # TODO - Might need some checks against the IRHS value, as listed in docstring
    fso = Float(0.0, low=0.0, high=6.0,
                     desc='Set this to 0.0 to let Overflow decide the default value.\n'
                          '1.0 - 1st-order spatial differencing of Euler terms.\n'
                          '2.0 - 2nd-order.\n'
                          '3.0 - 3rd-order.\n'
                          '4.0 - 4th-order.\n'
                          '5.0 - 5th-order.\n'
                          '6.0 - 6th-order.\n'
                          'Intermediate values allowed.\n'
                          'For IRHS=0, values of [2,6] are implemented: FSO=2 gives 2nd-order with 4/2 dissipation; FSO=3 gives 4th-order with 4/2; FSO=4 gives 4th-order with 6/2; FSO=5 gives 6th-order with 6/2; and FSO=6 gives 6th-order with 8/2.\n'
                          'For IRHS=2, values of [1,2] are implemented.\n'
                          'For IRHS=3-5, values of [1,3] are implemented; FSO>3 selects WENO5 or WENO5M. [2.0 for IRHS=0,2; 3.0 for IRHS=3-5]')
    delta = Float(1.0, 
                       desc='MUSCL scheme flux limiter flag:\n'
                            'For ILIMIT=1 (Koren limiter):\n'
                            '<0.0 - Turn off limiter.\n'
                            '0.0 - Koren limiter.\n'
                            '>0.0 - Koren limiter with CFL3D-type parameter epsilon=0.008*delta.\n'
                            'For ILIMIT=2-4 (minmod, van Albada, or WENO5M limiter):\n'
                            '<0.0 - Turn off limiter.\n'
                            '0.0-1.0 - Standard limiter implementation.\n'
                            '>1.0 - Added smoothing with pressure/temp switch (coefficient of DELTA-1).')
    filter = Enum(0, [0, 3, 5], 
                     desc='0 - No Q filtering.\n'
                          '3 - 3rd-order (5-point) Q filtering.\n'
                          '5 - 5th-order (7-point) Q filtering.\n'
                          'Filtering is only done for Newton/dual time-accurate runs.')
    epssgs = Float(0.02, 
                   desc='LU-SGS left-hand side spectral radius epsilon term (ILHS=3 only).')
    vepsl = Float(0.0, 
                  desc='Matrix dissipation minimum limit on linear eigenvalues.')
    vepsn = Float(0.0, 
                  desc='Matrix dissipation minimum limit on nonlinear eigenvalues.')
    roeavg = Bool(False, 
                  desc='Matrix dissipation flag to use Roe averaging for half-grid point flow quantities.')


class Namelist_VISINP(VariableTree):
    """VariableTree for the VISINP namelist. Viscous and turbulence modeling
    input. Contained in the Grid VariableTree.
    """

    # TODO - Funny default - True if Rey != 0.
    visc = Bool(False, 
                desc='True - Include all viscous terms including cross terms. This overrides VISCJ, VISCK, VISCL and VISCX.\n'
                     'False - Include only specified or automatically enabled viscous terms.')
    viscj = Bool(False, 
                 desc='True - Include viscous thin-layer terms in J.\n'
                      'False - Include viscous terms in J only if there are J-direction viscous walls.')
    visck = Bool(False, 
                 desc='True - Include viscous thin-layer terms in K.\n'
                      'False - Include viscous terms in K only if there are K-direction viscous walls.')
    viscl = Bool(False, 
                 desc='True - Include viscous thin-layer terms in L.\n'
                      'False - Include viscous terms in L only if there are L-direction viscous walls.')
    viscx = Bool(False, 
                 desc='True - Include viscous cross terms between coordinate directions that have thin-layer terms enabled.\n'
                      'False - No viscous cross terms.')
    wallfun = Bool(False, 
                   desc='True - Use wall function formulation for all viscous walls in this grid.\n'
                        'False - Use standard wall formulation.')
    cflt = Float(1.0, 
                 desc='Turbulence model time-step is CFLT times the flow solver time-step.')
    itert = Int(1, 
                desc='Number of turbulence model iterations per flow solver iteration (ITER); or number of turbulence model iterations per step if ITER=0.')
    # DONE - Special Defaults [3 for NQT=100-102; 1 for NQT=202-203; 10 for NQT=204-205]
    itlhit = Int(-1, 
                 desc='Number of subiterations for DDADI or SSOR scheme.\n'
                      'Set to -1 to let Overeflow decide the default value [3 for NQT=100-102; 1 for NQT=202-203; 10 for NQT=204-205])')
    # DONE - Special defaults [1.0 for 1-eq models; 2.0 for 2-eq models]
    fsot = Float(-1.0, 
                 desc='Set to 1.0 to let Overflow decide the defaults [1.0 for 1-eq models; 2.0 for 2-eq models]\n'
                      '1.0 - 1st-order differencing for turbulence convection terms.\n'
                      '2.0 - 2nd-order.\n'
                      '3.0 - 3rd-order.\n'
                      'Intermediate values allowed; values other than 1 are only implemented for 2-equation turbulence models.')
    mut_limit = Float(200000, 
                      desc='=0.0 - No limit on turbulent eddy viscosity.\n'
                           '>0.0 - Maximum limit for turbulent eddy viscosity.')
    ides = Enum(0, [0, 1, 2, 3], 
                desc='0 - No Detached Eddy Simulation (DES).\n'
                     '1 - Use original DES (applies to SA or SST models).\n'
                     '2 - Use delayed DES (DDES) (applies to SA or SST models).\n'
                     '3 - Use delayed Multi-Scale model (D-MS) (applies to SST; SA reverts to DDES).')
    irc = Enum(0, [0, 1, 2], 
               desc='0 - No rotational/curvature correction term for turbulence model.\n'
                    '1 - Use SARC form of rotational/curvature correction term.\n'
                    '2 - Use approximate rotational/curvature correction term.\n'
                    'May be applied to any 1- or 2-equation turbulence model.')
    icc = Enum(1, [0, 1], 
               desc='0 - No compressibility correction.\n'
                    '1 - Use Sarkar compressibility correction (SST model only).')
    itc = Enum(0, [0, 1], 
               desc='0 - No temperature correction.\n'
                    '1 - Use Abdol-Hamid temperature correction (2-equation models only).')
    ittyp = List([], Int(), desc='Turbulence modeling region type.')
    itdir = List([], Int(), 
                 desc='Turbulence model region coordinate direction (away from wall or shear layer). 1,2,3,-1,-2,-3 represent J,K,L,-J,-K,-L, resp.')
    jtls = List([], Int(), desc='Starting J index.')
    jtle = List([], Int(), desc='Ending J index.')
    ktls = List([], Int(), desc='Starting K index.')
    ktle = List([], Int(), desc='Ending K index.')
    ltls = List([], Int(), desc='Starting L index.')
    ltle = List([], Int(), desc='Ending L index.')
    # TODO - Float may not be right for this
    tlpar1 = List([], Float(), 
                  desc='Turbulence model region parameter (usage depends on region type).')
    

class Namelist_BCINP(VariableTree):
    """VariableTree for the BCINP namelist. Boundary condition input.
    Contained in the Grid VariableTree.
    """
    
    ibtyp = List([], Int(), desc='Boundary condition type.')
    ibdir = List([], Int(), 
                 desc='Boundary condition coordinate direction (away from boundary surface). 1,2,3,-1,-2,-3 represent J,K,L,-J,-K,-L, resp.')
    jbcs = List([], Int(), desc='Starting J index.')
    jbce = List([], Int(), desc='Ending J index.')
    kbcs = List([], Int(), desc='Starting K index.')
    kbce = List([], Int(), desc='Ending K index.')
    lbcs = List([], Int(), desc='Starting L index.')
    lbce = List([], Int(), desc='Ending L index.')
    bcpar1 = List([], Int(), 
                  desc='Boundary condition parameter (usage depends on boundary type).')
    bcpar2 = List([], Int(), 
                  desc='Boundary condition parameter (usage depends on boundary type).')
    bcfile = List([], Str(), 
                  desc='File name for reading boundary data (usage depends on boundary type).')
    
    
class Namelist_SCEINP(VariableTree):
    """VariableTree for the SCEINP namelist. Boundary condition input.
    Contained in the Grid VariableTree.
    """
    
    cflc = Float(1.0, 
                 desc='Species continuity equation time-step is CFLC times the flow solver time-step.')
    iterc = Int(1, 
                desc='Number of species continuity equation iterations per flow solver iteration (ITER); or number of species continuity equation iterations per step if ITER=0.')
    itlhic = Int(1, 
                 desc='Number of species equation left-hand side subiterations:\n'
                      '=1 - Use ADI left-hand side.\n'
                      '>1 - Use SSOR left-hand side.')
    iupc = Enum(1, [0, 1, 2], 
                desc='0 - Central differencing for species convection terms.\n'
                     '1 - Upwind differencing for species convection terms.\n'
                     '2 - HLLC upwind differencing for species convection terms.')
    # DONE - Strange defaults.
    fsoc = Float(0.0, low=0.0, high=3.0,
                 desc='Set to 0.0 to let Overflow determine the value from [2.0 for IUPC=0; 3.0 for IUPC=1-2]\n'
                      '1.0 - 1st-order differencing for species continuity terms.\n'
                      '2.0 - 2nd-order.\n'
                      '3.0 - 3rd-order.\n'
                      'Intermediate values allowed. For IUPC=0, only FSOC=2 is implemented; for IUPC=1-2, values of [1,3] are implemented.')
    dis2c = Float(2.0, desc='2nd-order smoothing coefficient.')
    dis4c = Float(0.04, desc='4th-order smoothing coefficient.')


class Namelist_SIXINP(VariableTree):
    """VariableTree for the SIXINP namelist. 6-DOF input.
    Contained in the Grid VariableTree.
    (OVERFLOW-D only; only for I6DOF!=2)
    """
    
    iblink = Int(1, desc='Body ID to which this grid is linked.')
    igmove = Enum(0, [0, 1], 
                  desc='0 - Body does not move (even if DYNMCS=TRUE).\n'
                       '1 - Body motion is enabled (if DYNMCS=TRUE).')
    bmass = Float(1.0, desc='Body mass.')
    tjj = Float(1.0, desc='Body momentsof inertia, about the principal axes (assumed to be body x,y,z).')
    tkk = Float(1.0, desc='Body momentsof inertia, about the principal axes (assumed to be body x,y,z).')
    tll = Float(1.0, desc='Body momentsof inertia, about the principal axes (assumed to be body x,y,z).')
    weight = Float(0.0, desc='Body weight.')
    gravx = Float(0.0, desc='Gravity unit vector (points in the direction of body weight).')
    gravy = Float(0.0, desc='Gravity unit vector (points in the direction of body weight).')
    gravz = Float(1.0, desc='Gravity unit vector (points in the direction of body weight).')
    ishift = Int(0, desc='Starting step number for applied loads (time=0).')
    fx = Float(0.0, desc='Body applied forces (in global x direction).')
    fy = Float(0.0, desc='Body applied forces (in global y direction).')
    fz = Float(0.0, desc='Body applied forces (in global z direction).')
    fmx = Float(0.0, desc='Body applied moments (about global x axis).')
    fmy = Float(0.0, desc='Body applied moments (about global y axis).')
    fmz = Float(0.0, desc='Body applied moments (about global z axis).')
    strokx = Float(0.0, desc='Translation of the body CG in x, defining the duration for applied loads to be active.')
    stroky = Float(0.0, desc='Translation of the body CG in y, defining the duration for applied loads to be active.')
    strokz = Float(0.0, desc='Translation of the body CG in z, defining the duration for applied loads to be active.')
    strokt = Float(0.0, desc='Time duration for applied loads to be active.')
    freex = Bool(True, desc='Enable/disable body movement in (x) directions (resp.), while applied loads are active.')
    freey = Bool(True, desc='Enable/disable body movement in (y) directions (resp.), while applied loads are active.')
    freez = Bool(True, desc='Enable/disable body movement in (z) directions (resp.), while applied loads are active.')
    freer = Bool(True, desc='Enable/disable (all 3) body rotational degrees-of-freedom, while applied loads are active.')
    free = Bool(False, desc='Enable/disable all body degrees-of-freedom, while applied loads are active (sets FREEX, FREEY, FREEZ, FREER).')
    x00 = Float(0.0, desc='Body CG location in body coordinates.')
    y00 = Float(0.0, desc='Body CG location in body coordinates.')
    z00 = Float(0.0, desc='Body CG location in body coordinates.')
    # TODO - default value for these next 3 is (x00, y00, z00)
    x0 = Float(0.0, desc='Initial body CG location in global coordinates.')
    y0 = Float(0.0, desc='Initial body CG location in global coordinates.')
    z0 = Float(0.0, desc='Initial body CG location in global coordinates.')
    e1 = Float(0.0, desc='Initial body Euler parameters in global coordinates.')
    e2 = Float(0.0, desc='Initial body Euler parameters in global coordinates.')
    e3 = Float(0.0, desc='Initial body Euler parameters in global coordinates.')
    e4 = Float(1.0, desc='Initial body Euler parameters in global coordinates.')
    ur = Float(0.0, desc='Initial velocity of CG in global coordinates.')
    vr = Float(0.0, desc='Initial velocity of CG in global coordinates.')
    wr = Float(0.0, desc='Initial velocity of CG in global coordinates.')
    wx = Float(0.0, desc='Initial angular velocity about CG in global coordinates.')
    wy = Float(0.0, desc='Initial angular velocity about CG in global coordinates.')
    wz = Float(0.0, desc='Initial angular velocity about CG in global coordinates.')
    wj = Float(0.0, desc='Initial angular velocity about CG in body coordinates.')
    wk = Float(0.0, desc='Initial angular velocity about CG in body coordinates.')
    wl = Float(0.0, desc='Initial angular velocity about CG in body coordinates.')


class Grid(VariableTree):
    """A VariableTree that holds all the grid-specific namelists for a given grid.
    This is functionally added to the OverflowWrapper.
    """

    name = Str("Grid", desc='A name for this grid')
    
    def __init__(self, *args, **kwargs):
        """Constructor for the Overflow wrapper"""

        super(Grid, self).__init__(*args, **kwargs)
        
        # Add VariableTrees
        self.add('niters',  Namelist_NITERS())
        self.add('metprm',  Namelist_METPRM())
        self.add('timacu',  Namelist_TIMACU())
        self.add('smoacu',  Namelist_SMOACU())
        self.add('visinp',  Namelist_VISINP())
        self.add('bcinp',  Namelist_BCINP())
        self.add('sceinp',  Namelist_SCEINP())
        self.add('sixinp',  Namelist_SIXINP())


class OverflowWrapper(ExternalCode):
    """ Wrapper for the Overflow CFD solver.
    """
    
    overflowD = Bool(True, iotype='in', desc='Set to True to run Overflow-D')
    error_text = Str('', iotype='out', desc='Error message(s) reported by Overflow')
    
    # Variable Tree slots
    global_params = Slot(Namelist_GLOBAL, iotype='in')
    omiglb = Slot(Namelist_OMIGLB, iotype='in')
    gbrick = Slot(Namelist_GBRICK, iotype='in')
    brkinp = Slot(Namelist_BRKINP, iotype='in')
    groups = Slot(Namelist_GROUPS, iotype='in')
    dcfglb = Slot(Namelist_DCFGLB, iotype='in')
    floinp = Slot(Namelist_FLOINP, iotype='in')
    vargam = Slot(Namelist_VARGAM, iotype='in')
    
    def __init__(self, *args, **kwargs):
        """Constructor for the Overflow wrapper"""
        
        super(OverflowWrapper, self).__init__(*args, **kwargs)

        self.stdout = 'over.out'
        self.stderr = 'over.err'
        self.command = 'overflow'
        
        self.inputfile = 'over.namelist'
        self.logfile = 'overflow.log'
        
        self.external_files = [
            FileMetadata(path='over.namelist', input=True),
            FileMetadata(path=self.stdout),
            FileMetadata(path=self.stderr),
        ]
        
        # Add VariableTrees
        self.add('global_params',  Namelist_GLOBAL())
        self.add('omiglb',  Namelist_OMIGLB())
        self.add('gbrick',  Namelist_GBRICK())
        self.add('brkinp',  Namelist_BRKINP())
        self.add('groups',  Namelist_GROUPS())
        self.add('dcfglb',  Namelist_DCFGLB())
        self.add('floinp',  Namelist_FLOINP())
        self.add('vargam',  Namelist_VARGAM())
        
        # Some defaults are dependent on other namelists.
        gam = self.floinp.gaminf
        self.vargam.alt0 = [gam/(gam-1.0)]
        
        # Private storage
        self._numgrids = 0
        
        
    def execute(self):
        """ do your calculations here """
        
        #Prepare the input file for Overflow
        self.generate_input()
        
        #Run Overflow via ExternalCode's execute function
        self.error_text = ''
        super(OverflowWrapper, self).execute()
        
        #Check for errors in the run
        self.check_errors()

        #Parse the outut file from Overflow
        self.parse_output()


    def generate_input(self):
        """Creates the namelist input file for Overflow."""
        
        sb = Namelist(self)
        sb.set_filename(self.inputfile)
        
        # Add the basic namelists
        
        sb.add_group('GLOBAL')
            
        skip = []
        if self.global_params.tphys == -9999:
            skip.append('tphys')
            
        sb.add_container('global_params', skip)
        
        if self.overflowD:
            
            sb.add_group('OMIGLB')
            sb.add_container('omiglb')
        
            sb.add_group('GBRICK')
            sb.add_container('gbrick')
        
            sb.add_group('BRKINP')
            sb.add_container('brkinp')
        
            sb.add_group('GROUPS')
            sb.add_container('groups')
        
            sb.add_group('DCFGLB')
            sb.add_container('dcfglb')
        
        sb.add_group('FLOINP')
        
        if self.floinp.refmach == 0:
            self.floinp.refmach = self.floinp.fsmach
            
        sb.add_container('floinp')
        
        sb.add_group('VARGAM')
        sb.add_container('vargam')
        
        # Add the Grid namelists
        
        for i in range(self._numgrids):
            
            stem = "Grid%d." % i
            
            sb.add_group('GRDNAM')
            sb.add_var(stem+'name')
            
            sb.add_group('NITERS')
            sb.add_container(stem+'niters')
            
            sb.add_group('METPRM')
            
            skip = []
            if self.get(stem+'metprm.ilhsit') == -1:
                skip.append('ilhsit')
                        
            sb.add_container(stem+'metprm', skip)
            
            sb.add_group('TIMACU')
            sb.add_container(stem+'timacu')
            
            sb.add_group('SMOACU')
            
            skip = []
            if self.get(stem+'smoacu.fso') == 0:
                skip.append('fso')
                        
            sb.add_container(stem+'smoacu', skip)
            
            sb.add_group('VISINP')
            
            skip = []
            if self.get(stem+'visinp.itlhit') == -1:
                skip.append('itlhit')
            if self.get(stem+'visinp.fsot') == -1:
                skip.append('fsot')
                        
            sb.add_container(stem+'visinp', skip)
            
            sb.add_group('BCINP')
            sb.add_container(stem+'bcinp')
            
            sb.add_group('SCEINP')
            
            skip = []
            if self.get(stem+'sceinp.fsoc') == 0:
                skip.append('fsoc')
                        
            sb.add_container(stem+'sceinp', skip)
            
            sb.add_group('SIXINP')
            sb.add_container(stem+'sixinp')
            
        # Generate the input file for Overflow
        sb.generate()
        
        
    def check_errors(self):
        """Checks for any errors in the run.
        
        Errors can be found in the logfile."""
        
    def parse_output(self):
        """Parses the Overflow output files and extracts data for the component
        outputs.
        Note, users will need to inherit and overload this method."""
    
        pass
    
    
    def load_model(self, filename):
        """Reads in an existing Overflow input file and populates the variable
        tree with its values."""

        sb = Namelist(self)
        sb.set_filename(filename)
        
        # Where each namelist goes in the component
        rule_dict = { "GLOBAL" : ["global_params"],
                      "OMIGLB" : ["omiglb"],
                      "GBRICK" : ["gbrick"],
                      "BRKINP" : ["brkinp"],
                      "GROUPS" : ["groups"],
                      "DCFGLB" : ["dcfglb"],
                      "FLOINP" : ["floinp"],
                      "VARGAM" : ["vargam"] }

        # Some variables aren't exposed in the OpenMDAO wrapper (e.g., array
        # sizes which aren't needed explicitly.)
        ignore = []
        
        sb.parse_file()
        empty_groups, unlisted_groups, unlinked_vars = \
                    sb.load_model(rule_dict, ignore)
        
        
        # Next, put every grid into a new Grid VariableTree.
        counter = 0
        in_grid = False
        for i, group in enumerate(sb.groups):
            
            # grdnam card means new grid coming up
            if group.lower().strip() == 'grdnam':
                
                container_name = "Grid%d" % counter
                counter += 1
                
                for item in ['NAME', 'Name', 'name']:
                    if sb.cards[i][0].name == item:
                        name = sb.cards[i][0].value
                        break
                else:
                    name = container_name
                    
                self.add_trait(container_name, Slot(Grid, iotype='in'))
                self.add(container_name, Grid())
                self.set('%s.name' % container_name, name)
                self._numgrids += 1
                
                in_grid = True
        
            # Every group that follows goes into current grid
            elif in_grid:
                
                stem = container_name+"."+group.lower()
                
                rule_dict = { group : [stem] }
                ignore = []
                
                # Start out with the default multig from globals, and let the
                # grids overwrite if they want.
                if group.lower() == 'metprm':
                    self.set(stem+'.multig', self.global_params.multig)
                
                ne, nu, nv = sb.load_model(rule_dict, ignore, i)

    def read_q(self, grid_file, q_file, multiblock=True, blanking=False, logger=None):
        """
        Read grid and solution files.
        Returns a :class:`DomainObj` initialized from `grid_file` and `q_file`.
    
        grid_file: string
            Grid filename.
    
        q_file: string
            Q data filename.
        """
        logger = logger or NullLogger()
    
        domain = read_plot3d_grid(grid_file, multiblock, dim=3, blanking=blanking,
                                  planes=False, binary=True, big_endian=False,
                                  single_precision=False, unformatted=True,
                                  logger=logger)
    
        with open(q_file, 'rb') as inp:
            logger.info("reading Q file '%s'", q_file)
            stream = Stream(inp, binary=True, big_endian=False,
                            single_precision=False, integer_8=False,
                            unformatted=True, recordmark_8=False)
            if multiblock:
                # Read number of zones.
                nblocks = stream.read_int(full_record=True)
            else:
                nblocks = 1
            if nblocks != len(domain.zones):
                raise RuntimeError('Q zones %d != Grid zones %d' \
                                   % (nblocks, len(domain.zones)))
    
            # Read zone dimensions, nq, nqc.
            reclen = stream.read_recordmark()
            expected = stream.reclen_ints(3*nblocks + 2)
            if reclen != expected:
                logger.warning('unexpected dimensions recordlength'
                               ' %d vs. %d', reclen, expected)
    
            for zone in domain.zones:
                name = domain.zone_name(zone)
                imax, jmax, kmax = stream.read_ints(3)
                if imax < 1 or jmax < 1 or kmax < 1:
                    raise ValueError("invalid dimensions: %dx%dx%d" \
                                     % (imax, jmax, kmax))
                logger.debug('    %s: %dx%dx%d', name, imax, jmax, kmax)
                zone_i, zone_j, zone_k = zone.shape
                if imax != zone_i or jmax != zone_j or kmax != zone_k:
                    raise RuntimeError('%s: Q %dx%dx%d != Grid %dx%dx%d' \
                                       % (name, imax, jmax, kmax,
                                          zone_i, zone_j, zone_k))
    
            nq, nqc = stream.read_ints(2)
            logger.debug('    nq %d, nqc %d', nq, nqc)
    
            reclen2 = stream.read_recordmark()
            if reclen2 != reclen:
                logger.warning('mismatched dimensions recordlength'
                               ' %d vs. %d', reclen2, reclen)
    
            # Read zone scalars and variables.
            for zone in domain.zones:
                name = domain.zone_name(zone)
                logger.debug('reading data for %s', name)
                self._read_scalars(zone, nqc, stream, logger)
                self._read_vars(zone, nq, nqc, stream, logger)
    
        return domain
    
    
    def _read_scalars(self, zone, nqc, stream, logger):
        """ Reads scalars for `zone`. """
        reclen = stream.read_recordmark()
        expected = stream.reclen_floats(7) \
                 + stream.reclen_ints(1) \
                 + stream.reclen_floats(3) \
                 + stream.reclen_floats(max(2, nqc)) \
                 + stream.reclen_floats(3)
        if reclen != expected:
            logger.warning('unexpected scalars recordlength'
                           ' %d vs. %d', reclen, expected)
    
        refmach, alpha, rey, time, gaminf, beta, tinf = stream.read_floats(7)
        igam = stream.read_int()
        htinf, ht1, ht2 = stream.read_floats(3)
        rgas = stream.read_floats(max(2, nqc)),
        fsmach, tvref, dtvref = stream.read_floats(3)
    
        logger.debug('    refmach %g, fsmach %r, alpha %g, beta %g, rey %g, time %g',
                     refmach, fsmach, alpha, beta, rey, time)
        logger.debug('    gaminf %g, igam %d, tinf %g, htinf %g, ht1 %g, ht2 %g',
                     gaminf, igam, tinf, htinf, ht1, ht2)
        logger.debug('    rgas %s', rgas)
        logger.debug('    tvref %g, dtvref %d', tvref, dtvref)
    
        flow = zone.flow_solution
        flow.refmach = refmach
        flow.fsmach = fsmach
        flow.alpha = alpha
        flow.beta = beta
        flow.rey = rey
        flow.time = time
        flow.gaminf = gaminf
        flow.igam = igam
        flow.tinf = tinf
        flow.htinf = htinf
        flow.ht1 = ht1
        flow.ht2 = ht2
        flow.rgas = rgas
        flow.tvref = tvref
        flow.dtvref = dtvref
    
        reclen2 = stream.read_recordmark()
        if reclen2 != reclen:
            logger.warning('mismatched dimensions recordlength'
                           ' %d vs. %d', reclen2, reclen)
    
    
    def _read_vars(self, zone, nq, nqc, stream, logger):
        """ Reads field variables for `zone` """
        shape = zone.shape
        imax, jmax, kmax = shape
        reclen = stream.read_recordmark()
        expected = stream.reclen_floats(nq * imax * jmax * kmax)
        if reclen != expected:
            logger.warning('unexpected Q variables recordlength'
                           ' %d vs. %d', reclen, expected)
        name = 'density'
        arr = stream.read_floats(shape, order='Fortran')
        logger.debug('    %s min %g, max %g', name, arr.min(), arr.max())
        zone.flow_solution.add_array(name, arr)
    
        vec = Vector()
    
        vec.x = stream.read_floats(shape, order='Fortran')
        logger.debug('    momentum.x min %g, max %g', vec.x.min(), vec.x.max())
    
        vec.y = stream.read_floats(shape, order='Fortran')
        logger.debug('    momentum.y min %g, max %g', vec.y.min(), vec.y.max())
    
        vec.z = stream.read_floats(shape, order='Fortran')
        logger.debug('    momentum.z min %g, max %g', vec.z.min(), vec.z.max())
    
        zone.flow_solution.add_vector('momentum', vec)
    
        name = 'energy_stagnation_density'
        arr = stream.read_floats(shape, order='Fortran')
        logger.debug('    %s min %g, max %g', name, arr.min(), arr.max())
        zone.flow_solution.add_array(name, arr)
    
        name = 'gamma'
        arr = stream.read_floats(shape, order='Fortran')
        logger.debug('    %s min %g, max %g', name, arr.min(), arr.max())
        zone.flow_solution.add_array(name, arr)
    
        for i in range(nqc):
            name = 'species_%d_density' % (i+1)
            arr = stream.read_floats(shape, order='Fortran')
            logger.debug('    %s min %g, max %g', name, arr.min(), arr.max())
            zone.flow_solution.add_array(name, arr)
    
        for i in range(nq - (6 + nqc)):
            name = 'turbulence_%d' % (i+1)
            arr = stream.read_floats(shape, order='Fortran')
            logger.debug('    %s min %g, max %g', name, arr.min(), arr.max())
            zone.flow_solution.add_array(name, arr)
    
        if stream.unformatted:
            reclen2 = stream.read_recordmark()
            if reclen2 != reclen:
                logger.warning('mismatched Q variables recordlength'
                               ' %d vs. %d', reclen2, reclen)
                
            
if __name__ == "__main__": # pragma: no cover     
    
    from openmdao.main.api import set_as_top
    from numpy import array
    
    my_comp = set_as_top(OverflowWrapper())

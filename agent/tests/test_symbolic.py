"""Unit tests for symbolic math and automatic differentiation tools."""

import asyncio

import numpy as np
import pytest

from huginn.tools.autodiff_tool import AutoDiffInput, AutoDiffTool
from huginn.tools.symbolic_math_tool import SymbolicMathInput, SymbolicMathTool
from huginn.types import ToolContext

CTX = ToolContext(session_id="test", workspace=".")


class TestSymbolicMathTool:
    @pytest.fixture
    def tool(self):
        return SymbolicMathTool()

    # ------------------------------------------------------------------
    # Basic calculus
    # ------------------------------------------------------------------
    def test_differentiate(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="differentiate",
                    expression="x**3 + 2*x**2",
                    symbols=["x"],
                    variable="x",
                    order=1,
                ),
                CTX,
            )
        )
        assert result.success
        assert "3*x**2" in result.data["result"].replace(" ", "")
        assert "latex" in result.data

    def test_integrate(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="integrate",
                    expression="2*x",
                    symbols=["x"],
                    variable="x",
                ),
                CTX,
            )
        )
        assert result.success
        assert "x**2" in result.data["result"]

    def test_solve(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="solve",
                    equations=["x**2 - 4 = 0"],
                    symbols=["x"],
                ),
                CTX,
            )
        )
        assert result.success
        assert len(result.data["solutions"]) == 2

    def test_simplify(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="simplify",
                    expression="sin(x)**2 + cos(x)**2",
                    symbols=["x"],
                ),
                CTX,
            )
        )
        assert result.success
        assert result.data["simplified"] == "1"

    def test_taylor(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="taylor",
                    expression="exp(x)",
                    symbols=["x"],
                    variable="x",
                    order=3,
                    point={"x": 0},
                ),
                CTX,
            )
        )
        assert result.success
        assert "x**3" in result.data["series"]

    def test_series(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="series",
                    expression="log(1 + x)",
                    symbols=["x"],
                    variable="x",
                    order=3,
                    point={"x": 0},
                ),
                CTX,
            )
        )
        assert result.success
        assert "x**3" in result.data["expansion"]

    # ------------------------------------------------------------------
    # Matrix / eigenvalue
    # ------------------------------------------------------------------
    def test_eigenvalue(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="eigenvalue",
                    matrix=[["a", "b"], ["b", "a"]],
                    symbols=["a", "b"],
                ),
                CTX,
            )
        )
        assert result.success
        assert len(result.data["eigenvalues"]) == 2

    # ------------------------------------------------------------------
    # Constitutive relations
    # ------------------------------------------------------------------
    def test_constitutive_stress_from_psi(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="constitutive",
                    free_energy="C10*(I1 - 3) + D1*(J - 1)**2",
                    symbols=["C10", "D1", "I1", "J", "C", "F"],
                    target="stress_from_psi",
                ),
                CTX,
            )
        )
        assert result.success
        assert "second_pk_stress" in result.data

    def test_constitutive_pressure_from_eos(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="constitutive",
                    free_energy="E0 + B0*V/BP * ((V0/V)**BP/(BP-1) + 1) - B0*V0/(BP-1)",
                    symbols=["E0", "B0", "V0", "BP", "V"],
                    target="pressure_from_eos",
                ),
                CTX,
            )
        )
        assert result.success
        assert "pressure" in result.data
        assert "bulk_modulus" in result.data

    def test_constitutive_chemical_potential(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="constitutive",
                    free_energy="mu0*n + R*T*n*log(n/V)",
                    symbols=["mu0", "n", "R", "T", "V"],
                    target="chemical_potential",
                ),
                CTX,
            )
        )
        assert result.success
        assert "chemical_potential" in result.data

    # ------------------------------------------------------------------
    # Weak form
    # ------------------------------------------------------------------
    def test_weak_form_derivation_1d(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="weak_form",
                    expression="-laplacian(u)",
                    symbols=["u", "v", "x"],
                    target="derivation",
                ),
                CTX,
            )
        )
        assert result.success
        assert "weak_form_terms" in result.data
        assert "diffusion" in result.data["weak_form_terms"]

    def test_weak_form_verification_2d(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="weak_form",
                    symbols=["u", "v", "x", "y"],
                    target="verification",
                ),
                CTX,
            )
        )
        assert result.success
        assert result.data["verified_symbolically"] is True

    def test_weak_form_linear_elasticity(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="weak_form",
                    symbols=["u", "v", "ux", "uy", "vx", "vy", "x", "y", "E", "nu"],
                    target="linear_elasticity",
                ),
                CTX,
            )
        )
        assert result.success
        assert result.data["element_type"] == "2D_plane_stress"
        assert "bilinear_form" in result.data
        assert "stiffness" in result.data["weak_form_terms"]

    def test_weak_form_heat_conduction(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="weak_form",
                    symbols=["u", "v", "x", "y", "k", "f"],
                    target="heat_conduction",
                ),
                CTX,
            )
        )
        assert result.success
        assert result.data["element_type"] == "heat_conduction"
        assert "bilinear_form" in result.data
        assert "diffusion" in result.data["weak_form_terms"]

    def test_weak_form_assemble_bar(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="weak_form",
                    expression="bar",
                    symbols=["u", "v", "x", "E", "A", "h"],
                    target="assemble_element_matrix",
                ),
                CTX,
            )
        )
        assert result.success
        assert result.data["element_type"] == "bar"
        assert result.data["size"] == 2
        assert result.data["is_symmetric"] is True
        K = result.data["element_matrix"]
        # When symbolic, entries are strings; when numeric, they are floats
        if isinstance(K[0][0], float):
            assert K[0][0] == -K[0][1]
        else:
            assert isinstance(K[0][0], str)
            assert "E" in K[0][0] and "A" in K[0][0]

    def test_weak_form_assemble_poisson_tri(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="weak_form",
                    expression="poisson_tri",
                    symbols=["u", "v", "x", "y", "k"],
                    target="assemble_element_matrix",
                ),
                CTX,
            )
        )
        assert result.success
        assert result.data["element_type"] == "poisson_tri"
        assert result.data["size"] == 3
        assert result.data["is_symmetric"] is True

    def test_weak_form_assemble_elasticity_tri(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="weak_form",
                    expression="elasticity_tri",
                    symbols=["u", "v", "x", "y", "E", "nu"],
                    target="assemble_element_matrix",
                ),
                CTX,
            )
        )
        assert result.success
        assert result.data["element_type"] == "elasticity_tri"
        assert result.data["size"] == 6
        assert result.data["is_symmetric"] is True

    # ------------------------------------------------------------------
    # Tensor ops
    # ------------------------------------------------------------------
    def test_tensor_ops_matrix(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="tensor_ops",
                    expression="A",
                    symbols=["A"],
                    matrix=[["a", "0"], ["0", "b"]],
                ),
                CTX,
            )
        )
        assert result.success
        assert "invariants" in result.data

    def test_tensor_ops_scalar(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="tensor_ops",
                    expression="(x + y)**2",
                    symbols=["x", "y"],
                ),
                CTX,
            )
        )
        assert result.success
        assert "expanded" in result.data

    # ------------------------------------------------------------------
    # Linear algebra
    # ------------------------------------------------------------------
    def test_linear_algebra_lu(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="linear_algebra",
                    target="lu_decompose",
                    matrix=[["4", "1"], ["1", "3"]],
                ),
                CTX,
            )
        )
        assert result.success
        assert "L" in result.data
        assert "U" in result.data
        assert result.data["size"] == 2

    def test_linear_algebra_cholesky(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="linear_algebra",
                    target="cholesky",
                    matrix=[["4", "1"], ["1", "3"]],
                ),
                CTX,
            )
        )
        assert result.success
        assert "L" in result.data
        assert result.data["size"] == 2

    def test_linear_algebra_jacobi(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="linear_algebra",
                    target="jacobi_solve",
                    matrix=[["4", "1"], ["1", "3"]],
                    expression="1,2",
                ),
                CTX,
            )
        )
        assert result.success
        assert result.data["solver"] == "jacobi_solve"
        assert len(result.data["solution"]) == 2

    def test_linear_algebra_cg(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="linear_algebra",
                    target="cg_solve",
                    matrix=[["4", "1"], ["1", "3"]],
                    expression="1,2",
                ),
                CTX,
            )
        )
        assert result.success
        assert result.data["solver"] == "cg_solve"
        assert len(result.data["solution"]) == 2

    def test_linear_algebra_mat_vec_mul(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="linear_algebra",
                    target="mat_vec_mul",
                    matrix=[["2", "0"], ["0", "3"]],
                    expression="1,1",
                ),
                CTX,
            )
        )
        assert result.success
        assert len(result.data["result"]) == 2

    def test_linear_algebra_cond_number(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="linear_algebra",
                    target="cond_number",
                    matrix=[["1", "0"], ["0", "100"]],
                ),
                CTX,
            )
        )
        assert result.success
        assert "cond_number" in result.data

    # ------------------------------------------------------------------
    # DFT
    # ------------------------------------------------------------------
    def test_dft_fermi_energy(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="dft",
                    target="fermi_energy",
                    expression="n=0.05",
                ),
                CTX,
            )
        )
        assert result.success
        assert "fermi_energy" in result.data
        assert result.data["fermi_energy"] > 0.0

    def test_dft_particle_in_box(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="dft",
                    target="particle_in_box",
                    expression="L=10.0,N=3",
                ),
                CTX,
            )
        )
        assert result.success
        assert len(result.data["levels"]) == 3
        assert result.data["levels"][0]["energy"] < result.data["levels"][1]["energy"]

    def test_dft_tight_binding_band(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="dft",
                    target="tight_binding_band",
                    expression="epsilon0=0.0,t=1.0,a=1.0,nK=10",
                ),
                CTX,
            )
        )
        assert result.success
        assert len(result.data["band"]) == 10
        assert result.data["band"][0]["energy"] >= result.data["band"][5]["energy"]

    def test_dft_lda_xc(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="dft",
                    target="lda_xc_energy",
                    expression="n=0.05",
                ),
                CTX,
            )
        )
        assert result.success
        assert "xc_energy_density" in result.data
        assert result.data["xc_energy_density"] < 0.0

    # ------------------------------------------------------------------
    # Thermodynamics
    # ------------------------------------------------------------------
    def test_thermo_ideal_gas(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="thermodynamics",
                    target="ideal_gas",
                    expression="n=1.0,T=273.15,V=0.022414",
                ),
                CTX,
            )
        )
        assert result.success
        assert "pressure" in result.data
        assert result.data["pressure"] > 100000.0

    def test_thermo_van_der_waals(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="thermodynamics",
                    target="van_der_waals",
                    expression="n=1.0,T=273.15,V=0.022414,a=0.364,b=4.27e-5",
                ),
                CTX,
            )
        )
        assert result.success
        assert "pressure" in result.data
        assert "critical_temperature" in result.data

    def test_thermo_helmholtz(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="thermodynamics",
                    target="helmholtz_energy",
                    expression="n=1.0,T=300.0,V1=1.0,V2=2.0",
                ),
                CTX,
            )
        )
        assert result.success
        assert "helmholtz_energy" in result.data

    def test_thermo_clausius_clapeyron(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="thermodynamics",
                    target="clausius_clapeyron",
                    expression="T=373.15,L=40700.0,deltaV=18.0e-6",
                ),
                CTX,
            )
        )
        assert result.success
        assert "slope_dPdT" in result.data
        assert result.data["slope_dPdT"] > 0.0

    def test_thermo_partition_function(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="thermodynamics",
                    target="partition_function",
                    expression="m=9.11e-31,T=300.0,V=1.0",
                ),
                CTX,
            )
        )
        assert result.success
        assert "single_partition_function" in result.data
        assert result.data["single_partition_function"] > 0.0

    # ------------------------------------------------------------------
    # Probability
    # ------------------------------------------------------------------
    def test_probability_normal_pdf(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="probability",
                    target="normal_pdf",
                    expression="mu=0.0,sigma=1.0,x=0.0",
                ),
                CTX,
            )
        )
        assert result.success
        assert "pdf" in result.data
        assert result.data["pdf"] > 0.38

    def test_probability_normal_cdf(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="probability",
                    target="normal_cdf",
                    expression="mu=0.0,sigma=1.0,x=0.0",
                ),
                CTX,
            )
        )
        assert result.success
        assert "cdf" in result.data
        assert 0.49 < result.data["cdf"] < 0.51

    def test_probability_gp_kernel(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="probability",
                    target="gp_kernel",
                    expression="sigma=1.0,lengthscale=1.0,x1=0.0,x2=1.0",
                    equations=["rbf"],
                ),
                CTX,
            )
        )
        assert result.success
        assert "kernel_value" in result.data
        assert result.data["kernel_value"] < 1.0

    def test_probability_monte_carlo(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="probability",
                    target="monte_carlo_integral",
                    expression="a=0.0,b=1.0,n=100",
                ),
                CTX,
            )
        )
        assert result.success
        assert "integral" in result.data
        assert result.data["integral"] > 0.3

    def test_probability_bayesian_update(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="probability",
                    target="bayesian_update_normal",
                    expression="mu0=0.0,tau0=1.0,sigma=0.25,data_mean=2.0,n=10.0",
                ),
                CTX,
            )
        )
        assert result.success
        assert "posterior_mean" in result.data
        assert result.data["posterior_mean"] > 1.9

    # ------------------------------------------------------------------
    # Dimensional analysis
    # ------------------------------------------------------------------
    def test_dimensional_analysis_check_equation(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="dimensional_analysis",
                    expression="210 GPa = 500 MPa / 0.001",
                    target="check_equation",
                ),
                CTX,
            )
        )
        assert result.success
        assert result.data["consistent"] is True

    def test_dimensional_analysis_buckingham_pi(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="dimensional_analysis",
                    expression="E:GPa, rho:g/cm3, L:m, v:m/s",
                    target="buckingham_pi",
                ),
                CTX,
            )
        )
        assert result.success
        assert len(result.data["pi_groups"]) > 0

    def test_dimensional_analysis_validate_expression(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="dimensional_analysis",
                    expression="stress = 500 MPa + 210 GPa * 0.001",
                    target="validate_expression",
                ),
                CTX,
            )
        )
        assert result.success
        assert len(result.data["quantities"]) > 0

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------
    def test_unknown_action(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(action="unknown"),
                CTX,
            )
        )
        assert not result.success
        assert "Unknown action" in result.error


class TestAutoDiffTool:
    @pytest.fixture
    def tool(self):
        return AutoDiffTool()

    # ------------------------------------------------------------------
    # Gradient
    # ------------------------------------------------------------------
    def test_gradient_birch_murnaghan(self, tool):
        result = asyncio.run(
            tool.call(
                AutoDiffInput(
                    action="gradient",
                    function_type="birch_murnaghan",
                    variables={"V": [100.0]},
                    function_params={"E0": 0.0, "B0": 100.0, "V0": 100.0, "BP": 4.0},
                    use_jax=False,  # ensure finite-difference path is covered
                ),
                CTX,
            )
        )
        assert result.success
        assert "gradients" in result.data
        assert "V" in result.data["gradients"]

    # ------------------------------------------------------------------
    # Hessian
    # ------------------------------------------------------------------
    def test_hessian_neo_hookean(self, tool):
        result = asyncio.run(
            tool.call(
                AutoDiffInput(
                    action="hessian",
                    function_type="neo_hookean",
                    variables={"I1": [3.0], "J": [1.0]},
                    function_params={"C10": 0.5, "D1": 2.0},
                    use_jax=tool._jax_available,
                ),
                CTX,
            )
        )
        if not tool._jax_available:
            pytest.skip("JAX not installed")
        assert result.success
        assert "hessian_matrix" in result.data
        assert "eigenvalues" in result.data
        assert isinstance(result.data["positive_definite"], bool)

    # ------------------------------------------------------------------
    # Jacobian
    # ------------------------------------------------------------------
    def test_jacobian_lennard_jones(self, tool):
        result = asyncio.run(
            tool.call(
                AutoDiffInput(
                    action="jacobian",
                    function_type="lennard_jones",
                    variables={"r": [1.5]},
                    function_params={"epsilon": 1.0, "sigma": 1.0},
                    use_jax=False,
                ),
                CTX,
            )
        )
        assert result.success
        assert "jacobian" in result.data
        assert "r" in result.data["jacobian"]

    # ------------------------------------------------------------------
    # Sensitivity
    # ------------------------------------------------------------------
    def test_sensitivity_morse(self, tool):
        result = asyncio.run(
            tool.call(
                AutoDiffInput(
                    action="sensitivity",
                    function_type="morse",
                    variables={"r": [1.2]},
                    function_params={"De": 1.0, "a": 1.0, "re": 1.0},
                    use_jax=False,
                ),
                CTX,
            )
        )
        assert result.success
        assert "sensitivities" in result.data
        assert "r" in result.data["sensitivities"]

    # ------------------------------------------------------------------
    # Optimize (previously had Python 3 zip indexing bug)
    # ------------------------------------------------------------------
    def test_optimize_birch_murnaghan(self, tool):
        # Synthetic data around V0=20, E0=-10
        V_data = [18.0, 19.0, 20.0, 21.0, 22.0]
        # Birch-Murnaghan with B0=100, V0=20, BP=4, E0=-10
        E_data = []
        for V in V_data:
            f = (V / 20.0) ** (-1.0 / 3.0) - 1.0
            E = -10.0 + 100.0 * 20.0 / 4.0 * (f**4 * (4 - 1) + 1) * np.exp(-f)
            E_data.append(E)

        result = asyncio.run(
            tool.call(
                AutoDiffInput(
                    action="optimize",
                    function_type="birch_murnaghan",
                    variables={"V": V_data, "target": E_data},
                    function_params={"E0": -8.0, "B0": 80.0, "V0": 18.0, "BP": 4.0},
                    use_jax=False,
                ),
                CTX,
            )
        )
        assert result.success
        assert "optimized_params" in result.data
        # Should move closer to true values
        opt = result.data["optimized_params"]
        assert opt["V0"] > 19.0  # moved toward 20
        assert result.data["final_loss"] < 1e6

    # ------------------------------------------------------------------
    # Error handling
    # ------------------------------------------------------------------
    def test_optimize_no_target(self, tool):
        result = asyncio.run(
            tool.call(
                AutoDiffInput(
                    action="optimize",
                    function_type="birch_murnaghan",
                    variables={"V": [1.0, 2.0]},
                    function_params={"E0": 0.0},
                    use_jax=False,
                ),
                CTX,
            )
        )
        assert not result.success
        assert "target" in result.error.lower()

    def test_unknown_action(self, tool):
        result = asyncio.run(
            tool.call(
                AutoDiffInput(action="unknown"),
                CTX,
            )
        )
        assert not result.success
        assert "Unknown action" in result.error


class TestUnifiedSymbolicMath:
    """Tests for symbolic_math_tool unified-framework bridge."""

    @pytest.fixture
    def tool(self):
        return SymbolicMathTool()

    def test_unified_list(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(action="unified", target="list"),
                CTX,
            )
        )
        assert result.success
        assert "models" in result.data
        assert "harmonic_oscillator_md" in result.data["models"]

    def test_unified_derive_harmonic(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="unified",
                    target="derive",
                    expression="harmonic_oscillator_md",
                ),
                CTX,
            )
        )
        assert result.success
        assert result.data["model"] == "harmonic_oscillator_md"
        assert result.data["principle"] == "hamiltonian"
        assert "energy_expression" in result.data
        assert "equations" in result.data
        eqs = result.data["equations"]
        assert "dq_dt" in eqs
        assert "dp_dt" in eqs

    def test_unified_derive_fem(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="unified",
                    target="derive",
                    expression="linear_elasticity_fem",
                ),
                CTX,
            )
        )
        assert result.success
        assert result.data["principle"] == "minimum"
        assert "displacement" in result.data["equations"]

    def test_unified_bridge_dft_to_md(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="unified",
                    target="bridge",
                    expression="dft-to-md",
                ),
                CTX,
            )
        )
        assert result.success
        assert result.data["bridge"] == "dft_to_md"
        assert result.data["potential_name"]

    def test_unified_unknown_model(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="unified",
                    target="derive",
                    expression="nonexistent_model",
                ),
                CTX,
            )
        )
        assert not result.success
        assert "Unknown unified model" in result.error

    def test_unified_bridge_md_to_stress(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="unified",
                    target="bridge",
                    expression="md-to-stress",
                ),
                CTX,
            )
        )
        assert result.success
        assert result.data["bridge"] == "md_to_stress"
        assert "cauchy_stress" in result.data["result"]

    def test_unified_bridge_md_to_elasticity(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="unified",
                    target="bridge",
                    expression="md-to-elasticity",
                    free_energy="0.5*k*(r-r0)**2",
                    symbols=["k", "r", "r0"],
                ),
                CTX,
            )
        )
        assert result.success
        assert result.data["bridge"] == "md_to_elasticity"
        assert "elastic_modulus" in result.data["result"]

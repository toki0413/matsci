import asyncio

from huginn.core_types import ToolContext
from huginn.tools.symbolic_math import SymbolicMathInput, SymbolicMathTool


async def main():
    t = SymbolicMathTool()

    # 1) classify: NLSE i u_t + u_xx + 2|u|^2 u = 0
    #    second-order part: A u_xx + 2B u_xy + C u_yy  -> A=1, B=0, C=0
    r = await t.call(
        SymbolicMathInput(action="pde_classify", expression="1;0;0", symbols=["x", "t"]),
        ToolContext(session_id="test", workspace="/workspace/agent"),
    )
    print("CLASSIFY:", r.data, r.success, r.error)

    # 2) noether: Lagrangian for NLSE.  Use real-field form.
    r2 = await t.call(
        SymbolicMathInput(
            action="noether",
            expression="u_x^2 - u_t^2/2",
            symbols=["u", "x", "t"],
            target="translation",
        ),
        ToolContext(session_id="test", workspace="/workspace/agent"),
    )
    print("NOETHER:", r2.data, r2.success, r2.error)


asyncio.run(main())
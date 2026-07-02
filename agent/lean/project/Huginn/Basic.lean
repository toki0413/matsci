import Mathlib

namespace Huginn

/-- A material system has a state space and an evolution function. -/
class MaterialSystem (α : Type) where
  state_space : Type
  evolution : α → α

/-- A conservation law asserts an invariant under evolution. -/
class ConservationLaw (M : MaterialSystem α) where
  invariant : α → ℝ
  preserved : ∀ s, invariant (M.evolution s) = invariant s

end Huginn

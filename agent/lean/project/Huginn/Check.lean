import Huginn.Basic


import Huginn.Basic

noncomputable section

def myEvolution (s : ℝ) : ℝ := s

instance : MaterialSystem ℝ where
  state_space := ℝ
  evolution := myEvolution

instance : ConservationLaw (MaterialSystem.mk ℝ myEvolution) where
  invariant := id
  preserved := by intro s; simp [myEvolution]

end


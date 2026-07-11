/**
 * useTeam — Multi-agent team planning and execution state.
 *
 * Calls /team/v2/* API endpoints (ModelTeam with multi-model routing).
 * Falls back to /team/* (legacy Orchestrator) if v2 unavailable.
 */
import { useState } from "react";
import { api } from "../lib/api";

export function useTeam() {
  const [teamObjective, setTeamObjective] = useState("");
  const [teamPlan, setTeamPlan] = useState<any[] | null>(null);
  const [teamRunning, setTeamRunning] = useState(false);
  const [teamResult, setTeamResult] = useState<any>(null);
  const [teamError, setTeamError] = useState("");

  const handleTeamPlan = async () => {
    if (!teamObjective.trim()) return;
    setTeamRunning(true);
    setTeamError("");
    setTeamResult(null);
    try {
      // Try v2 API (ModelTeam) first
      let data = await api.post<{ success?: boolean; tasks?: any[]; plan?: any[]; error?: string }>(
        "/team/v2/plan",
        { objective: teamObjective }
      ).catch(() => null);

      // Fallback to legacy API
      if (!data || !data.success) {
        data = await api.post<{ success?: boolean; tasks?: any[]; error?: string }>(
          "/team/plan",
          { objective: teamObjective }
        );
      }

      if (data.success) {
        setTeamPlan(data.tasks || data.plan || []);
      } else {
        setTeamError(data.error || "Planning failed.");
        setTeamPlan(null);
      }
    } catch (e: any) {
      setTeamError(e.message || "Network error");
    } finally {
      setTeamRunning(false);
    }
  };

  const handleTeamRun = async () => {
    if (!teamObjective.trim()) return;
    setTeamRunning(true);
    setTeamError("");
    setTeamResult(null);
    try {
      // Try v2 API (ModelTeam) first
      let data = await api.post<{ success?: boolean; error?: string } & Record<string, any>>(
        "/team/v2/run",
        { objective: teamObjective }
      ).catch(() => null);

      // Fallback to legacy API
      if (!data || !data.success) {
        data = await api.post<{ success?: boolean; error?: string } & Record<string, any>>(
          "/team/run",
          { objective: teamObjective }
        );
      }

      if (data.success) {
        setTeamResult(data);
      } else {
        setTeamError(data.error || "Team run failed.");
      }
    } catch (e: any) {
      setTeamError(e.message || "Network error");
    } finally {
      setTeamRunning(false);
    }
  };

  const [teamFusionResult, setTeamFusionResult] = useState<any>(null);

  const handleTeamFusion = async (rounds: number = 1) => {
    if (!teamObjective.trim()) return;
    setTeamRunning(true);
    setTeamError("");
    setTeamFusionResult(null);
    try {
      const data = await api.post<{ success?: boolean; error?: string } & Record<string, any>>(
        "/team/v2/fusion",
        { query: teamObjective, rounds }
      );
      if (data.success) {
        setTeamFusionResult(data);
      } else {
        setTeamError(data.error || "Fusion failed.");
      }
    } catch (e: any) {
      setTeamError(e.message || "Network error");
    } finally {
      setTeamRunning(false);
    }
  };

  return {
    teamObjective, teamPlan, teamRunning, teamResult, teamFusionResult, teamError,
    setTeamObjective,
    handleTeamPlan, handleTeamRun, handleTeamFusion,
  };
}

/**
 * useTeam — Multi-agent team planning and execution state.
 *
 * Manages team objective, plan tasks, running state, results and errors.
 * Calls /team/plan and /team/run API endpoints.
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
      const data = await api.post<{ success?: boolean; tasks?: any[]; error?: string }>(
        "/team/plan",
        { objective: teamObjective }
      );
      if (data.success) {
        setTeamPlan(data.tasks || []);
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
      const data = await api.post<{ success?: boolean; error?: string } & Record<string, any>>(
        "/team/run",
        { objective: teamObjective }
      );
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

  return {
    teamObjective, teamPlan, teamRunning, teamResult, teamError,
    setTeamObjective,
    handleTeamPlan, handleTeamRun,
  };
}

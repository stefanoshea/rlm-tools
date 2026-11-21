import { apiClient, clock, logger } from "./deps";

export interface DashboardRendering {
  render(): string;
}

export class DashboardReducer implements DashboardRendering {
  apiClient = apiClient;
  clock = clock;

  render(): string {
    return logger(`dashboard:${this.clock.now()}`);
  }
}

export class SettingsReducer {
  logger = logger;

  render(): string {
    return this.logger(apiClient.get("/settings"));
  }
}

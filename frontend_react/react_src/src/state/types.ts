export type ValueOf<T> = T[keyof T];

export interface AppStateData {
  showWelcome: boolean;
  collapsiblePanelDefaultOpen: boolean;
  enableChartGeneration: boolean;
  enableBusinessInsights: boolean;
  dataSource: string;
  expandGraphsInsightsDefaultOpen: boolean;
  theme: 'light' | 'dark';
}

export interface AppStateActions {
  hideWelcomeModal: () => void;
  setCollapsiblePanelDefaultOpen: (isOpen: boolean) => void;
  setEnableChartGeneration: (enabled: boolean) => void;
  setEnableBusinessInsights: (enabled: boolean) => void;
  setDataSource: (source: string) => void;
  setExpandGraphsInsightsDefaultOpen: (isOpen: boolean) => void;
  setTheme: (theme: 'light' | 'dark') => void;
}

export type AppState = AppStateData & AppStateActions;

export type Action = 
  | { type: "HIDE_WELCOME_MODAL" }
  | { type: "SET_COLLAPSIBLE_PANEL_DEFAULT_OPEN"; payload: boolean }
  | { type: "SET_ENABLE_CHART_GENERATION"; payload: boolean }
  | { type: "SET_ENABLE_BUSINESS_INSIGHTS"; payload: boolean }
  | { type: "SET_DATA_SOURCE"; payload: string }
  | { type: "SET_EXPAND_GRAPHS_INSIGHTS_DEFAULT_OPEN"; payload: boolean }
  | { type: "SET_THEME"; payload: 'light' | 'dark' };

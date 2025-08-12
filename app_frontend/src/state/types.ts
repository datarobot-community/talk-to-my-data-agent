export type ValueOf<T> = T[keyof T];

export interface AppStateData {
  showWelcome: boolean;
  collapsiblePanelDefaultOpen: boolean;
  enableChartGeneration: boolean;
  enableBusinessInsights: boolean;
  includeCsvBom: boolean;
  dataSource: string;
}

export interface AppStateActions {
  hideWelcomeModal: () => void;
  setCollapsiblePanelDefaultOpen: (isOpen: boolean) => void;
  setEnableChartGeneration: (enabled: boolean) => void;
  setEnableBusinessInsights: (enabled: boolean) => void;
  setIncludeCsvBom: (enabled: boolean) => void;
  setDataSource: (source: string) => void;
}

export type AppState = AppStateData & AppStateActions;

export type Action =
  | { type: 'HIDE_WELCOME_MODAL' }
  | { type: 'SET_COLLAPSIBLE_PANEL_DEFAULT_OPEN'; payload: boolean }
  | { type: 'SET_ENABLE_CHART_GENERATION'; payload: boolean }
  | { type: 'SET_ENABLE_BUSINESS_INSIGHTS'; payload: boolean }
  | { type: 'SET_INCLUDE_CSV_BOM'; payload: boolean }
  | { type: 'SET_DATA_SOURCE'; payload: string };

interface IMetadata {
  duration?: number | null;
  attempts?: number | null;
  datasets_analyzed?: number | null;
  total_rows_analyzed?: number | null;
  total_columns_analyzed?: number | null;
  exception?: {
    exception_history?: ICodeExecutionError[] | null;
  } | null;
  columns_analyzed?: number;
  rows_analyzed?: number;
  question?: string;
}

export interface ICodeExecutionError {
  code?: string | null;
  exception_str?: string | null;
  stdout?: string | null;
  stderr?: string | null;
  traceback_str?: string | null;
}

export interface IDataset {
  name: string;
  data_records: Record<string, unknown>[];
}

export interface IMessageComponent {
  enhanced_user_message?: string;
}

export interface IAnalysisComponent extends IComponent {
  type: 'analysis';
  metadata?: IMetadata;
  dataset_id?: string | null;
  code?: string | null;
}

export interface IChartsComponent extends IComponent {
  type: 'charts';
  fig1_json?: string | null;
  fig2_json?: string | null;
  code?: string | null;
}

export interface IBusinessComponent extends IComponent {
  type: 'business';
  bottom_line?: string | null;
  additional_insights?: string | null;
  follow_up_questions?: string[] | null;
}

export interface ITokenUsageInfo {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  call_count: number;
  model: string;
}

export interface IUsageInfoComponent {
  type: 'usage_info';
  usage: ITokenUsageInfo;
}

interface IComponent {
  status?: 'success' | 'error';
  metadata?: IMetadata;
}

export interface IChatMessageStep {
  step: string;
  reattempt: number;
}

export interface IChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  components: (
    | IMessageComponent
    | IAnalysisComponent
    | IChartsComponent
    | IBusinessComponent
    | IUsageInfoComponent
  )[];
  step?: IChatMessageStep;
  in_progress?: boolean;
  created_at?: string; // ISO timestamp for message creation time
  chat_id?: string; // ID of the chat this message belongs to
  id?: string; // Unique identifier for the message
  error?: string;
}

export interface IUserMessage {
  message: string;
  chatId?: string;
  enableChartGeneration?: boolean;
  enableBusinessInsights?: boolean;
  dataSource?: string;
}

export interface IPostMessageContext {
  previousMessages: IChatMessage[];
  messagesKey: string[];
  previousChats?: IChat[];
}

export interface IChat {
  id: string;
  name: string;
  created_at: string; // ISO date for chat creation time
  data_source?: string;
}

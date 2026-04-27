export type DictionaryTable = {
  name: string;
  column_descriptions?: Array<DictionaryRow>;
  in_progress: boolean;
  error?: string | null;
};

export type DictionaryRow = {
  column: string;
  data_type: string;
  description: string;
};

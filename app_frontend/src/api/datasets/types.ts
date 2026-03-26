export type Dataset = {
  id: string;
  name: string;
  created: string;
  size: string;
  file_size?: number;
};

export type DatasetResponse = {
  dataset: {
    name: string;
    data_records: Record<string, unknown>[];
  };
  cleaning_report?: unknown[];
  dataset_name?: string;
};

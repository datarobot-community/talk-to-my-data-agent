// eslint-disable-next-line @typescript-eslint/no-explicit-any
export const localizeException = (t: (a: string) => string, error: any) => {
  if (error?.response?.data?.detail?.code) {
    const code = error.response.data.detail.code;
    switch (code) {
      case 'DATASET_USED':
        return t('Dataset in use.');
      case 'DATASET_TOO_LARGE':
        return t('Datasets exceed maximum size.');
      case 'DATASET_INVALID':
        return t('The dataset cannot be used.');
    }
  }
  return null;
};

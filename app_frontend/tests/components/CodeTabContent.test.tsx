import { screen } from '@testing-library/react';
import { test, describe, expect, vi, afterEach } from 'vitest';
import { CodeTabContent } from '@/components/chat/CodeTabContent';
import { renderWithProviders } from '../test-utils';

vi.mock('@/api/datasets/hooks', () => ({
  useDownloadDataset: () => ({ mutate: vi.fn(), isPending: false }),
}));

describe('CodeTabContent provenance strip', () => {
  afterEach(() => vi.clearAllMocks());

  test('hides strip when usedDatasets is missing or empty', () => {
    const { rerender } = renderWithProviders(
      <CodeTabContent code="print('hi')" datasetId={null} />
    );
    expect(screen.queryByTestId('provenance-strip')).not.toBeInTheDocument();

    rerender(<CodeTabContent code={null} datasetId={null} usedDatasets={[]} />);
    expect(screen.queryByTestId('provenance-strip')).not.toBeInTheDocument();
  });

  test('renders strip with comma-separated names', () => {
    renderWithProviders(
      <CodeTabContent
        code={null}
        datasetId={null}
        usedDatasets={['sales_q1', 'sales_q2', 'returns']}
      />
    );

    expect(screen.getByTestId('provenance-strip')).toHaveTextContent('sales_q1, sales_q2, returns');
  });
});

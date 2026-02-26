import { render, screen } from '@testing-library/react';
import { test, describe, expect, vi } from 'vitest';
import { DatasetCardActionBar } from '@/components/data/DatasetCardActionBar';
import { DATA_TABS } from '@/state/constants';

describe('DatasetCardActionBar', () => {
  test('renders download button when onDownload is provided', () => {
    render(<DatasetCardActionBar onDownload={vi.fn()} viewMode={DATA_TABS.DESCRIPTION} />);
    expect(screen.getByTestId('download-dictionary-button')).toBeInTheDocument();
  });

  test('does not render download button when onDownload is undefined', () => {
    render(<DatasetCardActionBar viewMode={DATA_TABS.RAW} />);
    expect(screen.queryByTestId('download-dictionary-button')).not.toBeInTheDocument();
  });
});

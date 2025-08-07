import { render, screen } from '@testing-library/react';
import { test, describe, expect, vi } from 'vitest';
import userEvent from '@testing-library/user-event';
import { MessageHeader } from '@/components/chat/MessageHeader';

describe('MessageHeader Component', () => {
  const defaultProps = {
    name: 'Test User',
    date: '2024-01-01',
  };

  test('renders basic header information', () => {
    render(<MessageHeader {...defaultProps} />);

    expect(screen.getByText('Test User')).toBeInTheDocument();
    expect(screen.getByText('2024-01-01')).toBeInTheDocument();
  });

  test('renders delete button when onDelete is provided', () => {
    const onDelete = vi.fn();
    render(<MessageHeader {...defaultProps} onDelete={onDelete} />);

    const deleteButton = screen.getByRole('button', { name: /delete message and response/i });
    expect(deleteButton).toBeInTheDocument();
  });

  test('renders export button and calls onExport when clicked', async () => {
    const user = userEvent.setup();
    const onExport = vi.fn();
    render(<MessageHeader {...defaultProps} onExport={onExport} />);

    const exportButton = screen.getByRole('button', { name: /export chat/i });
    expect(exportButton).toBeInTheDocument();

    await user.click(exportButton);

    expect(onExport).toHaveBeenCalledTimes(1);
  });

  test('shows confirm dialog when delete button is clicked', async () => {
    const user = userEvent.setup();
    const onDelete = vi.fn();
    render(<MessageHeader {...defaultProps} onDelete={onDelete} />);

    const deleteButton = screen.getByRole('button', { name: /delete message and response/i });
    await user.click(deleteButton);

    expect(screen.getByText('Delete message')).toBeInTheDocument();
    expect(screen.getByText('Are you sure you want to delete this message?')).toBeInTheDocument();
  });

  test('calls onDelete when confirm dialog is confirmed', async () => {
    const user = userEvent.setup();
    const onDelete = vi.fn();
    render(<MessageHeader {...defaultProps} onDelete={onDelete} />);

    const deleteButton = screen.getByRole('button', { name: /delete message and response/i });
    await user.click(deleteButton);

    const confirmButton = screen.getByTestId('confirm-dialog-confirm');
    await user.click(confirmButton);

    expect(onDelete).toHaveBeenCalledTimes(1);
  });

  test('closes confirm dialog when cancel is clicked', async () => {
    const user = userEvent.setup();
    const onDelete = vi.fn();
    render(<MessageHeader {...defaultProps} onDelete={onDelete} />);

    const deleteButton = screen.getByRole('button', { name: /delete message and response/i });
    await user.click(deleteButton);

    const cancelButton = screen.getByTestId('confirm-dialog-cancel');
    await user.click(cancelButton);

    expect(screen.queryByText('Delete message')).not.toBeInTheDocument();
    expect(onDelete).not.toHaveBeenCalled();
  });

  test('disables export button when isExporting is true', () => {
    const onExport = vi.fn();
    render(<MessageHeader {...defaultProps} onExport={onExport} isExporting={true} />);

    const exportButton = screen.getByRole('button', { name: /exporting/i });
    expect(exportButton).toBeDisabled();
  });

  test('disables export button when response is in progress', () => {
    const onExport = vi.fn();
    render(<MessageHeader {...defaultProps} onExport={onExport} isResponseInProgress={true} />);

    const exportButton = screen.getByRole('button', {
      name: /Wait for agent to finish responding/i,
    });
    expect(exportButton).toBeDisabled();
  });

  test('disables export button when response is failing', () => {
    const onExport = vi.fn();
    render(<MessageHeader {...defaultProps} onExport={onExport} isResponseFailing={true} />);

    const exportButton = screen.getByRole('button', { name: /cannot export chat with errors/i });
    expect(exportButton).toBeDisabled();
    expect(exportButton).toHaveAttribute('title', 'Cannot export chat with errors');
  });

  test('shows correct tooltip when response is in progress', () => {
    const onExport = vi.fn();
    render(<MessageHeader {...defaultProps} onExport={onExport} isResponseInProgress={true} />);

    const exportButton = screen.getByRole('button', {
      name: /Wait for agent to finish responding/i,
    });
    expect(exportButton).toHaveAttribute('title', 'Wait for agent to finish responding');
  });
});

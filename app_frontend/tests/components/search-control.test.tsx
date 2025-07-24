import React from 'react';
import '@testing-library/jest-dom';
import { render, screen } from '@testing-library/react';
import { test, describe, expect, vi } from 'vitest';
import userEvent from '@testing-library/user-event';
import { SearchControl } from '@/components/ui-custom/search-control';

describe('SearchControl Component', () => {
  test('renders collapsed search button by default', () => {
    render(<SearchControl />);
    const searchButton = screen.getByTestId('search-control-button');

    expect(searchButton).toBeInTheDocument();
    expect(screen.getByText('Search')).toBeInTheDocument();
  });

  test('expands when search button is clicked', async () => {
    const user = userEvent.setup();
    render(<SearchControl />);

    const searchButton = screen.getByTestId('search-control-button');
    await user.click(searchButton);

    const searchInput = screen.getByTestId('search-control-input');
    expect(searchInput).toBeInTheDocument();
    expect(searchInput).toHaveFocus();
  });

  test('calls onSearch when input value changes', async () => {
    const mockOnSearch = vi.fn();
    const user = userEvent.setup();
    render(<SearchControl onSearch={mockOnSearch} />);

    // Expand the search
    const searchButton = screen.getByTestId('search-control-button');
    await user.click(searchButton);

    // Type in the input
    const searchInput = screen.getByTestId('search-control-input');
    await user.type(searchInput, 'test query');

    expect(mockOnSearch).toHaveBeenCalledWith('test query');
  });

  test('shows clear button when input has text', async () => {
    const user = userEvent.setup();
    render(<SearchControl />);

    // Expand and type
    const searchButton = screen.getByTestId('search-control-button');
    await user.click(searchButton);

    const searchInput = screen.getByTestId('search-control-input');
    await user.type(searchInput, 'test');

    const clearButton = screen.getByTestId('search-control-clear');
    expect(clearButton).toBeInTheDocument();
  });

  test('clears input and calls onSearch when clear button is clicked', async () => {
    const mockOnSearch = vi.fn();
    const user = userEvent.setup();
    render(<SearchControl onSearch={mockOnSearch} />);

    // Expand and type
    const searchButton = screen.getByTestId('search-control-button');
    await user.click(searchButton);

    const searchInput = screen.getByTestId('search-control-input');
    await user.type(searchInput, 'test');

    // Clear the input
    const clearButton = screen.getByTestId('search-control-clear');
    await user.click(clearButton);

    // After clearing, the component should collapse back to button state
    expect(screen.getByTestId('search-control-button')).toBeInTheDocument();
    expect(screen.queryByTestId('search-control-input')).not.toBeInTheDocument();
    expect(mockOnSearch).toHaveBeenCalledWith('');
  });

  test('collapses when input loses focus and is empty', async () => {
    const user = userEvent.setup();
    render(<SearchControl />);

    // Expand the search
    const searchButton = screen.getByTestId('search-control-button');
    await user.click(searchButton);

    // Verify input is present
    expect(screen.getByTestId('search-control-input')).toBeInTheDocument();

    // Focus out without typing
    await user.click(document.body);

    // Should collapse back to button
    expect(screen.getByTestId('search-control-button')).toBeInTheDocument();
    expect(screen.queryByTestId('search-control-input')).not.toBeInTheDocument();
  });

  test('does not collapse when input loses focus but has text', async () => {
    const user = userEvent.setup();
    render(<SearchControl />);

    // Expand and type
    const searchButton = screen.getByTestId('search-control-button');
    await user.click(searchButton);

    const searchInput = screen.getByTestId('search-control-input');
    await user.type(searchInput, 'test');

    // Focus out
    await user.click(document.body);

    // Should stay expanded
    expect(screen.getByTestId('search-control-input')).toBeInTheDocument();
  });

  test('clears input and collapses when Escape key is pressed', async () => {
    const mockOnSearch = vi.fn();
    const user = userEvent.setup();
    render(<SearchControl onSearch={mockOnSearch} />);

    // Expand and type
    const searchButton = screen.getByTestId('search-control-button');
    await user.click(searchButton);

    const searchInput = screen.getByTestId('search-control-input');
    await user.type(searchInput, 'test');

    // Press Escape
    await user.keyboard('{Escape}');

    expect(mockOnSearch).toHaveBeenCalledWith('');
    expect(screen.getByTestId('search-control-button')).toBeInTheDocument();
    expect(screen.queryByTestId('search-control-input')).not.toBeInTheDocument();
  });

  test('respects disabled prop', async () => {
    const user = userEvent.setup();
    render(<SearchControl disabled />);

    const searchButton = screen.getByTestId('search-control-button');
    expect(searchButton).toBeDisabled();

    // Should not expand when clicked
    await user.click(searchButton);
    expect(screen.queryByTestId('search-control-input')).not.toBeInTheDocument();
  });

  test('uses custom placeholder', async () => {
    const user = userEvent.setup();
    render(<SearchControl placeholder="Custom placeholder" />);

    // Expand the search
    const searchButton = screen.getByTestId('search-control-button');
    await user.click(searchButton);

    const searchInput = screen.getByTestId('search-control-input');
    expect(searchInput).toHaveAttribute('placeholder', 'Custom placeholder');
  });

  test('does not show clear button when disabled', async () => {
    const user = userEvent.setup();
    const { rerender } = render(<SearchControl disabled={false} />);

    // Expand and type
    const searchButton = screen.getByTestId('search-control-button');
    await user.click(searchButton);

    const searchInput = screen.getByTestId('search-control-input');
    await user.type(searchInput, 'test');

    // Simulate disabling after typing
    rerender(<SearchControl disabled />);

    // The clear button should not be present when disabled
    expect(screen.queryByTestId('search-control-clear')).not.toBeInTheDocument();
  });
});

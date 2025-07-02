import { render, screen } from '@testing-library/react';
import { test, describe, expect, vi } from 'vitest';
import userEvent from '@testing-library/user-event';
import { Input } from '@/components/ui/input';

describe('Input Component', () => {
  test('renders input with default props', () => {
    render(<Input />);
    const input = screen.getByRole('textbox');
    
    expect(input).toBeInTheDocument();
    expect(input).toHaveClass('h-9');
  });

  test('sets the correct input type', () => {
    const { rerender } = render(<Input type="text" />);
    let input = screen.getByRole('textbox');
    expect(input).toHaveAttribute('type', 'text');
    
    rerender(<Input type="email" />);
    input = screen.getByRole('textbox');
    expect(input).toHaveAttribute('type', 'email');
    
    rerender(<Input type="password" />);
    expect(screen.queryByRole('textbox')).not.toBeInTheDocument();
    const passwordInput = screen.getByDisplayValue('');
    expect(passwordInput).toHaveAttribute('type', 'password');
  });

  test('applies custom className', () => {
    render(<Input className="test-class" />);
    const input = screen.getByRole('textbox');
    
    expect(input).toHaveClass('test-class');
  });

  test('accepts and displays value correctly', async () => {
    const user = userEvent.setup();
    render(<Input defaultValue="Default text" />);
    const input = screen.getByRole('textbox');
    
    expect(input).toHaveValue('Default text');
    
    await user.clear(input);
    await user.type(input, 'New text');
    
    expect(input).toHaveValue('New text');
  });

  test('handles disabled state correctly', async () => {
    const user = userEvent.setup();
    render(<Input disabled />);
    const input = screen.getByRole('textbox');
    
    expect(input).toBeDisabled();
    
    await user.type(input, 'Test text');
    expect(input).not.toHaveValue('Test text');
  });

  test('passes additional props to the input element', async () => {
    const handleChange = vi.fn();
    const user = userEvent.setup();
    render(<Input onChange={handleChange} placeholder="Enter text here" />);
    const input = screen.getByPlaceholderText('Enter text here');
    
    await user.type(input, 'a');
    expect(handleChange).toHaveBeenCalledTimes(1);
  });

  test('handles aria-invalid attribute', () => {
    render(<Input aria-invalid={true} />);
    const input = screen.getByRole('textbox');
    
    expect(input).toHaveAttribute('aria-invalid', 'true');
    expect(input).toHaveClass('aria-invalid:border-destructive/60');
  });
});

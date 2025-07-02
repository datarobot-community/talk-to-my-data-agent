import * as React from "react";
import { render, screen, fireEvent } from '@testing-library/react';
import { test, describe, expect, vi } from 'vitest';
import userEvent from '@testing-library/user-event';
import { PromptInput } from '@/components/ui-custom/prompt-input';

// Create a mock icon component for testing
const MockIcon = (props:React.ComponentProps<React.ElementType>) => <div data-testid="mock-icon" {...props}>Icon</div>;

describe('PromptInput Component', () => {
  test('renders input with default props', () => {
    render(<PromptInput iconProps={{ behavior: 'append' }} />);
    
    const input = screen.getByRole('textbox');
    expect(input).toBeInTheDocument();
    expect(input).toHaveClass('bg-transparent');
  });

  test('handles focus and blur states correctly', async () => {
    render(<PromptInput iconProps={{ behavior: 'append' }} />);
    
    const input = screen.getByRole('textbox');
    const container = input.parentElement;
    
    // Initially not focused
    expect(container).not.toHaveClass('ring-4');
    
    // Focus the input
    fireEvent.focus(input);
    expect(container).toHaveClass('ring-4');
    
    // Blur the input
    fireEvent.blur(input);
    expect(container).not.toHaveClass('ring-4');
  });

  test('renders prepend icon correctly', () => {
    render(
      <PromptInput 
        icon={MockIcon} 
        iconProps={{ behavior: 'prepend', 'data-custom': 'test-value' }} 
      />
    );
    
    const icon = screen.getByTestId('mock-icon');
    expect(icon).toBeInTheDocument();
    expect(icon.parentElement).toHaveClass('mr-3');
    expect(icon).toHaveAttribute('data-custom', 'test-value');
  });

  test('renders append icon correctly', () => {
    render(
      <PromptInput 
        icon={MockIcon} 
        iconProps={{ behavior: 'append', 'data-custom': 'test-value' }} 
      />
    );
    
    const icon = screen.getByTestId('mock-icon');
    expect(icon).toBeInTheDocument();
    expect(icon.parentElement).toHaveClass('ml-3');
    expect(icon).toHaveAttribute('data-custom', 'test-value');
  });

  test('passes input props correctly', async () => {
    const handleChange = vi.fn();
    const user = userEvent.setup();
    render(
      <PromptInput 
        iconProps={{ behavior: 'append' }} 
        placeholder="Enter text"
        onChange={handleChange}
        disabled
      />
    );
    
    const input = screen.getByPlaceholderText('Enter text');
    expect(input).toBeDisabled();
    
    await user.type(input, 'Hello');
    expect(handleChange).not.toHaveBeenCalled(); // Because it's disabled
  });

  test('applies custom class names', () => {
    render(
      <PromptInput 
        iconProps={{ behavior: 'append' }} 
        className="custom-class"
      />
    );
    
    const input = screen.getByRole('textbox');
    const container = input.parentElement;
    
    expect(input).toHaveClass('custom-class');
    expect(container).toHaveClass('custom-class');
  });

  test('forwards ref to input element', () => {
    const ref = vi.fn();
    render(<PromptInput iconProps={{ behavior: 'append' }} ref={ref} />);
    
    expect(ref).toHaveBeenCalled();
  });
});

import { render, screen, fireEvent } from '@testing-library/react';
import { vi } from 'vitest';
import { PromptInput } from '@/components/ui-custom/prompt-input';

describe('PromptInput', () => {
  it('renders correctly', () => {
    render(<PromptInput />);
    expect(screen.getByRole('textbox')).toBeInTheDocument();
  });

  it('applies custom class name', () => {
    render(<PromptInput className="custom-class" />);
    expect(screen.getByRole('textbox')).toHaveClass('custom-class');
  });

  it('renders with send button appended by default', () => {
    render(<PromptInput testId="test-prompt-input-container" />);
    expect(screen.getByTestId('send-message-button')).toBeInTheDocument();
    // Check the main container for the default flex-row class
    const container = screen.getByTestId('test-prompt-input-container');
    expect(container).toHaveClass('flex-row');
  });

  it('renders with send button prepended', () => {
    render(<PromptInput sendButtonArrangement="prepend" />);
    expect(screen.getByTestId('send-message-button')).toBeInTheDocument();
    // Verify the order, assuming the button is the first child
    expect(screen.getByTestId('send-message-button').previousElementSibling).toBeNull();
  });

  it('disables send button when processing is true', () => {
    render(<PromptInput isProcessing={true} />);
    expect(screen.getByTestId('send-message-button')).toBeDisabled();
  });

  it('calls onSend when send button is clicked', () => {
    const handleSend = vi.fn();
    render(<PromptInput onSend={handleSend} initialValue="test message" />);
    fireEvent.click(screen.getByTestId('send-message-button'));
    expect(handleSend).toHaveBeenCalledTimes(1);
    expect(handleSend).toHaveBeenCalledWith('test message');
  });

  it('calls onSend when Enter key is pressed', () => {
    const handleSend = vi.fn();
    render(<PromptInput onSend={handleSend} initialValue="test message" />);
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter', code: 'Enter' });
    expect(handleSend).toHaveBeenCalledTimes(1);
    expect(handleSend).toHaveBeenCalledWith('test message');
  });

  it('does not call onSend when Enter is pressed with Shift', () => {
    const handleSend = vi.fn();
    render(<PromptInput onSend={handleSend} initialValue="test message" />);
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter', code: 'Enter', shiftKey: true });
    expect(handleSend).not.toHaveBeenCalled();
  });

  it('disables input and send button when isDisabled is true', () => {
    const handleSend = vi.fn();
    render(<PromptInput onSend={handleSend} isDisabled={true} />);
    const textarea = screen.getByRole('textbox');
    const sendButton = screen.getByTestId('send-message-button');

    expect(textarea).toBeDisabled();
    expect(sendButton).toBeDisabled();

    fireEvent.keyDown(textarea, { key: 'Enter', code: 'Enter' });
    expect(handleSend).not.toHaveBeenCalled();
    fireEvent.click(sendButton);
    expect(handleSend).not.toHaveBeenCalled();
  });

  it('does not call onSend when processing is true', () => {
    const handleSend = vi.fn();
    render(<PromptInput onSend={handleSend} isProcessing={true} initialValue="test message" />);
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter', code: 'Enter' });
    expect(handleSend).not.toHaveBeenCalled();
  });

  it('manages message state internally', () => {
    render(<PromptInput initialValue="initial text" />);
    const textarea = screen.getByRole('textbox');
    expect(textarea).toHaveValue('initial text');

    fireEvent.change(textarea, { target: { value: 'updated text' } });
    expect(textarea).toHaveValue('updated text');
  });

  it('clears message after sending', () => {
    const handleSend = vi.fn();
    render(<PromptInput onSend={handleSend} initialValue="test message" />);

    const textarea = screen.getByRole('textbox');
    expect(textarea).toHaveValue('test message');

    fireEvent.click(screen.getByTestId('send-message-button'));
    expect(handleSend).toHaveBeenCalledWith('test message');
    expect(textarea).toHaveValue('');
  });

  it('disables send button when message is empty', () => {
    render(<PromptInput />);
    const sendButton = screen.getByTestId('send-message-button');
    expect(sendButton).toBeDisabled();
  });

  it('enables send button when message has content', () => {
    render(<PromptInput initialValue="test" />);
    const sendButton = screen.getByTestId('send-message-button');
    expect(sendButton).not.toBeDisabled();
  });

  it('disables send button when message is only whitespace', () => {
    render(<PromptInput initialValue="   " />);
    const sendButton = screen.getByTestId('send-message-button');
    expect(sendButton).toBeDisabled();
  });

  it('does not call onSend when message is empty', () => {
    const handleSend = vi.fn();
    render(<PromptInput onSend={handleSend} />);

    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter', code: 'Enter' });
    expect(handleSend).not.toHaveBeenCalled();

    fireEvent.click(screen.getByTestId('send-message-button'));
    expect(handleSend).not.toHaveBeenCalled();
  });

  it('does not call onSend when message is only whitespace', () => {
    const handleSend = vi.fn();
    render(<PromptInput onSend={handleSend} initialValue="   " />);

    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter', code: 'Enter' });
    expect(handleSend).not.toHaveBeenCalled();

    fireEvent.click(screen.getByTestId('send-message-button'));
    expect(handleSend).not.toHaveBeenCalled();
  });
});

describe('PromptInput Tooltip', () => {
  it('shows "Processing..." tooltip when isProcessing is true', () => {
    render(<PromptInput isProcessing={true} />);
    const sendButtonSpan = screen.getByTestId('send-message-button').parentElement;
    expect(sendButtonSpan).toHaveAttribute('title', 'Processing... Waiting for response.');
  });

  it('shows "Ask a question" tooltip when isDisabled is true and not processing', () => {
    render(<PromptInput isDisabled={true} />);
    const sendButtonSpan = screen.getByTestId('send-message-button').parentElement;
    expect(sendButtonSpan).toHaveAttribute('title', 'Ask a question');
  });

  it('shows "Ask a question" tooltip when message is empty and not disabled/processing', () => {
    render(<PromptInput initialValue="" />);
    const sendButtonSpan = screen.getByTestId('send-message-button').parentElement;
    expect(sendButtonSpan).toHaveAttribute('title', 'Ask a question');
  });

  it('shows "Send message" tooltip when message has content and not disabled/processing', () => {
    render(<PromptInput initialValue="Hello" />);
    const sendButton = screen.getByTestId('send-message-button');
    expect(sendButton).toHaveAttribute('title', 'Send message');
  });
});

describe('PromptInput Icon Change', () => {
  it('shows faHourglassHalf icon when processing is true', () => {
    render(<PromptInput isProcessing={true} initialValue="test" />);
    const hourglassIcon = screen
      .getByTestId('send-message-button')
      .querySelector('[data-icon="hourglass-half"]');
    expect(hourglassIcon).toBeInTheDocument();
  });

  it('shows faPaperPlane icon when processing is false', () => {
    render(<PromptInput isProcessing={false} initialValue="test" />);
    const paperPlaneIcon = screen
      .getByTestId('send-message-button')
      .querySelector('[data-icon="paper-plane"]');
    expect(paperPlaneIcon).toBeInTheDocument();
  });
});

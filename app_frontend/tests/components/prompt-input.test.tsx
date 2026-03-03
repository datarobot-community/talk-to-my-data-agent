import { render, screen, fireEvent } from '@testing-library/react';
import { vi } from 'vitest';
import { PromptInput } from '@/components/ui-custom/prompt-input';
import { MAX_PROMPT_LENGTH } from '@/constants/chat';

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

describe('PromptInput Character Limit', () => {
  it('does not show counter below 80% of limit', () => {
    const shortText = 'a'.repeat(Math.ceil(MAX_PROMPT_LENGTH * 0.8) - 1);
    render(<PromptInput initialValue={shortText} />);
    expect(screen.queryByTestId('char-counter')).not.toBeInTheDocument();
  });

  it('shows character counter near the limit', () => {
    const nearLimitText = 'a'.repeat(Math.ceil(MAX_PROMPT_LENGTH * 0.8));
    render(<PromptInput initialValue={nearLimitText} />);
    const counter = screen.getByTestId('char-counter');
    expect(counter).toBeInTheDocument();
    expect(counter).toHaveTextContent(`${nearLimitText.length}/${MAX_PROMPT_LENGTH}`);
  });

  it('disables sending at exactly max length', () => {
    const maxText = 'a'.repeat(MAX_PROMPT_LENGTH);
    render(<PromptInput initialValue={maxText} />);
    const sendButton = screen.getByTestId('send-message-button');
    expect(sendButton).toBeDisabled();
  });

  it('shows "Message limit reached" and disables send over max length', () => {
    const overLimitText = 'a'.repeat(MAX_PROMPT_LENGTH + 1);
    render(<PromptInput initialValue={overLimitText} />);
    const counter = screen.getByTestId('char-counter');
    expect(counter).toHaveTextContent(`Message limit reached (${MAX_PROMPT_LENGTH} characters)`);
    expect(screen.getByTestId('send-message-button')).toBeDisabled();
  });

  it('does not send via Enter when over the limit', () => {
    const handleSend = vi.fn();
    const overLimitText = 'a'.repeat(MAX_PROMPT_LENGTH + 1);
    render(<PromptInput onSend={handleSend} initialValue={overLimitText} />);
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter', code: 'Enter' });
    expect(handleSend).not.toHaveBeenCalled();
  });

  it('does not send via Enter at exactly max length', () => {
    const handleSend = vi.fn();
    const maxText = 'a'.repeat(MAX_PROMPT_LENGTH);
    render(<PromptInput onSend={handleSend} initialValue={maxText} />);
    fireEvent.keyDown(screen.getByRole('textbox'), { key: 'Enter', code: 'Enter' });
    expect(handleSend).not.toHaveBeenCalled();
  });

  it('shows counter after typing past 80% threshold', () => {
    render(<PromptInput initialValue="" />);
    expect(screen.queryByTestId('char-counter')).not.toBeInTheDocument();
    const longText = 'a'.repeat(Math.ceil(MAX_PROMPT_LENGTH * 0.8));
    fireEvent.change(screen.getByTestId('prompt-input-textarea'), { target: { value: longText } });
    expect(screen.getByTestId('char-counter')).toBeInTheDocument();
  });
});

describe('PromptInput Icon Change', () => {
  it('shows Hourglass icon when processing is true', () => {
    render(<PromptInput isProcessing={true} initialValue="test" />);
    const button = screen.getByTestId('send-message-button');
    expect(button.querySelector('[data-testid="hourglass-icon"]')).toBeInTheDocument();
    expect(button.querySelector('[data-testid="send-icon"]')).not.toBeInTheDocument();
  });

  it('shows Send icon when processing is false', () => {
    render(<PromptInput isProcessing={false} initialValue="test" />);
    const button = screen.getByTestId('send-message-button');
    expect(button.querySelector('[data-testid="send-icon"]')).toBeInTheDocument();
    expect(button.querySelector('[data-testid="hourglass-icon"]')).not.toBeInTheDocument();
  });
});

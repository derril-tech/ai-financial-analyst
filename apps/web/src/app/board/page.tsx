'use client';

import { useState, useCallback } from 'react';
import {
  Container,
  Title,
  TextInput,
  Button,
  Grid,
  Card,
  Text,
  Badge,
  ActionIcon,
  Group,
  Stack,
  Loader,
  Alert,
} from '@mantine/core';
import { IconSearch, IconPlus, IconX, IconDownload, IconEye } from '@tabler/icons-react';
import { DragDropContext, Droppable, Draggable } from '@hello-pangea/dnd';
import { notifications } from '@mantine/notifications';

interface ResearchCard {
  id: string;
  type: 'query' | 'chart' | 'table' | 'valuation';
  title: string;
  content: any;
  citations?: Citation[];
  metadata?: Record<string, any>;
}

interface Citation {
  source: string;
  kind: 'pdf' | 'xbrl' | 'audio' | 'slide' | 'api';
  locator: string;
  confidence: number;
}

export default function ResearchBoardPage() {
  const [query, setQuery] = useState('');
  const [cards, setCards] = useState<ResearchCard[]>([]);
  const [loading, setLoading] = useState(false);

  const handleSearch = useCallback(async () => {
    if (!query.trim()) return;

    setLoading(true);
    try {
      // Mock API call - replace with actual API
      const response = await fetch('/api/v1/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          org_id: 'demo-org',
          prompt: query,
          tickers: [],
        }),
      });

      if (response.ok) {
        const result = await response.json();
        
        const newCard: ResearchCard = {
          id: `card-${Date.now()}`,
          type: 'query',
          title: query,
          content: {
            text: result.text || 'Analysis results would appear here...',
            confidence: result.confidence || 0.85,
          },
          citations: result.citations || [],
          metadata: {
            timestamp: new Date().toISOString(),
            query: query,
          },
        };

        setCards(prev => [...prev, newCard]);
        setQuery('');
        
        notifications.show({
          title: 'Analysis Complete',
          message: 'New research card added to your board',
          color: 'green',
        });
      } else {
        throw new Error('Failed to process query');
      }
    } catch (error) {
      console.error('Search error:', error);
      
      // Add mock card for demo purposes
      const mockCard: ResearchCard = {
        id: `card-${Date.now()}`,
        type: 'query',
        title: query,
        content: {
          text: `Mock analysis for: "${query}". This demonstrates the research board interface. In production, this would show real financial analysis with citations and data.`,
          confidence: 0.78,
        },
        citations: [
          {
            source: 'NVDA 10-K 2023',
            kind: 'pdf',
            locator: 'Page 45, Revenue Section',
            confidence: 0.92,
          },
          {
            source: 'Q3 2023 Earnings Call',
            kind: 'audio',
            locator: '15:30 - CEO Comments',
            confidence: 0.85,
          },
        ],
        metadata: {
          timestamp: new Date().toISOString(),
          query: query,
        },
      };

      setCards(prev => [...prev, mockCard]);
      setQuery('');
      
      notifications.show({
        title: 'Demo Mode',
        message: 'Added mock research card (API not connected)',
        color: 'blue',
      });
    } finally {
      setLoading(false);
    }
  }, [query]);

  const handleDragEnd = useCallback((result: any) => {
    if (!result.destination) return;

    const items = Array.from(cards);
    const [reorderedItem] = items.splice(result.source.index, 1);
    items.splice(result.destination.index, 0, reorderedItem);

    setCards(items);
  }, [cards]);

  const removeCard = useCallback((cardId: string) => {
    setCards(prev => prev.filter(card => card.id !== cardId));
  }, []);

  const addValuationCard = useCallback(() => {
    const valuationCard: ResearchCard = {
      id: `valuation-${Date.now()}`,
      type: 'valuation',
      title: 'DCF Valuation - NVDA',
      content: {
        enterprise_value: 2450000000000, // $2.45T
        equity_value: 2380000000000,     // $2.38T
        share_price: 485.20,
        current_price: 452.30,
        upside: 7.3,
        model: 'Three-Stage DCF',
        wacc: 0.095,
        terminal_growth: 0.025,
      },
      metadata: {
        timestamp: new Date().toISOString(),
        model_type: 'dcf',
      },
    };

    setCards(prev => [...prev, valuationCard]);
    
    notifications.show({
      title: 'Valuation Added',
      message: 'DCF model card added to research board',
      color: 'green',
    });
  }, []);

  return (
    <Container size="xl" py="xl">
      <Stack gap="lg">
        {/* Header */}
        <Group justify="space-between">
          <Title order={1}>Research Board</Title>
          <Group>
            <Button 
              leftSection={<IconPlus size={16} />}
              variant="light"
              onClick={addValuationCard}
            >
              Add Valuation
            </Button>
          </Group>
        </Group>

        {/* Search Bar */}
        <Card withBorder>
          <Group>
            <TextInput
              flex={1}
              placeholder="Ask about financial performance, ratios, growth trends..."
              value={query}
              onChange={(e) => setQuery(e.currentTarget.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
              leftSection={<IconSearch size={16} />}
            />
            <Button 
              onClick={handleSearch}
              loading={loading}
              disabled={!query.trim()}
            >
              Analyze
            </Button>
          </Group>
        </Card>

        {/* Research Cards */}
        {cards.length === 0 ? (
          <Alert title="Welcome to Research Board" color="blue">
            Start by asking a question about financial data, or add a valuation model. 
            Cards can be dragged to reorder and clicked to drill down into details.
          </Alert>
        ) : (
          <DragDropContext onDragEnd={handleDragEnd}>
            <Droppable droppableId="research-cards">
              {(provided) => (
                <div {...provided.droppableProps} ref={provided.innerRef}>
                  <Grid>
                    {cards.map((card, index) => (
                      <Draggable key={card.id} draggableId={card.id} index={index}>
                        {(provided, snapshot) => (
                          <Grid.Col
                            span={{ base: 12, md: 6, lg: 4 }}
                            ref={provided.innerRef}
                            {...provided.draggableProps}
                            {...provided.dragHandleProps}
                            style={{
                              ...provided.draggableProps.style,
                              opacity: snapshot.isDragging ? 0.8 : 1,
                            }}
                          >
                            <ResearchCardComponent 
                              card={card} 
                              onRemove={removeCard}
                            />
                          </Grid.Col>
                        )}
                      </Draggable>
                    ))}
                  </Grid>
                  {provided.placeholder}
                </div>
              )}
            </Droppable>
          </DragDropContext>
        )}
      </Stack>
    </Container>
  );
}

interface ResearchCardProps {
  card: ResearchCard;
  onRemove: (id: string) => void;
}

function ResearchCardComponent({ card, onRemove }: ResearchCardProps) {
  const [showDetails, setShowDetails] = useState(false);

  const getCardColor = (type: string) => {
    switch (type) {
      case 'query': return 'blue';
      case 'valuation': return 'green';
      case 'chart': return 'orange';
      case 'table': return 'purple';
      default: return 'gray';
    }
  };

  const renderCardContent = () => {
    switch (card.type) {
      case 'query':
        return (
          <Stack gap="sm">
            <Text size="sm" lineClamp={3}>
              {card.content.text}
            </Text>
            {card.content.confidence && (
              <Badge size="sm" color="blue">
                Confidence: {(card.content.confidence * 100).toFixed(0)}%
              </Badge>
            )}
          </Stack>
        );
      
      case 'valuation':
        return (
          <Stack gap="sm">
            <Group justify="space-between">
              <Text size="sm" fw={500}>Target Price</Text>
              <Text size="lg" fw={700} c="green">
                ${card.content.share_price?.toFixed(2)}
              </Text>
            </Group>
            <Group justify="space-between">
              <Text size="sm">Current Price</Text>
              <Text size="sm">${card.content.current_price?.toFixed(2)}</Text>
            </Group>
            <Group justify="space-between">
              <Text size="sm">Upside</Text>
              <Badge color={card.content.upside > 0 ? 'green' : 'red'}>
                {card.content.upside?.toFixed(1)}%
              </Badge>
            </Group>
            <Text size="xs" c="dimmed">
              {card.content.model} | WACC: {(card.content.wacc * 100)?.toFixed(1)}%
            </Text>
          </Stack>
        );
      
      default:
        return <Text size="sm">Card content</Text>;
    }
  };

  return (
    <Card withBorder h={280} style={{ display: 'flex', flexDirection: 'column' }}>
      {/* Card Header */}
      <Group justify="space-between" mb="sm">
        <Badge size="sm" color={getCardColor(card.type)}>
          {card.type.toUpperCase()}
        </Badge>
        <Group gap="xs">
          <ActionIcon 
            size="sm" 
            variant="subtle"
            onClick={() => setShowDetails(!showDetails)}
          >
            <IconEye size={14} />
          </ActionIcon>
          <ActionIcon 
            size="sm" 
            variant="subtle"
            color="red"
            onClick={() => onRemove(card.id)}
          >
            <IconX size={14} />
          </ActionIcon>
        </Group>
      </Group>

      {/* Card Title */}
      <Text fw={500} size="sm" lineClamp={2} mb="sm">
        {card.title}
      </Text>

      {/* Card Content */}
      <div style={{ flex: 1 }}>
        {renderCardContent()}
      </div>

      {/* Citations */}
      {card.citations && card.citations.length > 0 && (
        <Group gap="xs" mt="sm">
          <Text size="xs" c="dimmed">Sources:</Text>
          {card.citations.slice(0, 2).map((citation, idx) => (
            <Badge key={idx} size="xs" variant="light">
              {citation.source}
            </Badge>
          ))}
          {card.citations.length > 2 && (
            <Badge size="xs" variant="light">
              +{card.citations.length - 2} more
            </Badge>
          )}
        </Group>
      )}

      {/* Details Panel */}
      {showDetails && (
        <Card withBorder mt="sm" p="xs">
          <Stack gap="xs">
            <Text size="xs" fw={500}>Details</Text>
            {card.citations?.map((citation, idx) => (
              <Group key={idx} justify="space-between">
                <Text size="xs">{citation.source}</Text>
                <Badge size="xs" color="blue">
                  {(citation.confidence * 100).toFixed(0)}%
                </Badge>
              </Group>
            ))}
            {card.metadata?.timestamp && (
              <Text size="xs" c="dimmed">
                Created: {new Date(card.metadata.timestamp).toLocaleString()}
              </Text>
            )}
          </Stack>
        </Card>
      )}
    </Card>
  );
}
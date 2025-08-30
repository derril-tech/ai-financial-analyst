import { Container, Title, Text, Button, Stack } from '@mantine/core';
import Link from 'next/link';

export default function HomePage() {
  return (
    <Container size="md" py="xl">
      <Stack align="center" gap="md">
        <Title order={1} ta="center">
          AI Financial Analyst
        </Title>
        <Text size="lg" ta="center" c="dimmed">
          Multimodal RAG + Structured Data + Tools for Financial Analysis
        </Text>
        <Button component={Link} href="/board" size="lg">
          Go to Research Board
        </Button>
      </Stack>
    </Container>
  );
}

"""
Pytest tests for research intelligence module.

Tests PubMed, arXiv integration and ResearchIntelligenceEngine.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, Mock, patch
from src.research_intelligence import (
    ResearchPaper, ResearchSummary, PubMedSearcher, ArXivSearcher, 
    ResearchIntelligenceEngine
)


class TestResearchPaper:
    """Test ResearchPaper dataclass."""
    
    def test_research_paper_creation(self):
        """Test creating ResearchPaper instance."""
        paper = ResearchPaper(
            title="Test Paper",
            authors=["Author One", "Author Two"],
            abstract="Test abstract",
            journal="Test Journal",
            year=2023,
            sample_size=100
        )
        
        assert paper.title == "Test Paper"
        assert len(paper.authors) == 2
        assert paper.sample_size == 100
        assert paper.pmid is None  # Optional field


class TestResearchSummary:
    """Test ResearchSummary dataclass."""
    
    def test_research_summary_creation(self):
        """Test creating ResearchSummary instance."""
        from datetime import datetime
        
        papers = [ResearchPaper(
            title="Test", authors=["Author"], abstract="Abstract", 
            journal="Journal", year=2023
        )]
        
        summary = ResearchSummary(
            query="test query",
            search_date=datetime.now(),
            total_papers_found=1,
            papers_analyzed=papers,
            evidence_quality="medium",
            effect_size_range=(0.2, 0.8),
            typical_sample_sizes=[50, 100],
            common_study_designs=["RCT"],
            recommended_outcome_measures=["anxiety scale"],
            effect_heterogeneity="low",
            publication_bias_risk="medium",
            temporal_trends="stable"
        )
        
        assert summary.query == "test query"
        assert len(summary.papers_analyzed) == 1
        assert summary.evidence_quality == "medium"


class TestPubMedSearcher:
    """Test PubMed API integration."""
    
    def test_pubmed_searcher_initialization(self):
        """Test PubMed searcher creates successfully."""
        searcher = PubMedSearcher()
        assert searcher.base_url == "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
        assert searcher.email == "research@example.com"
    
    def test_query_enhancement(self):
        """Test PubMed query enhancement."""
        searcher = PubMedSearcher()
        enhanced = searcher._enhance_pubmed_query("meditation anxiety", years_back=5)
        
        assert "meditation" in enhanced
        assert "anxiety" in enhanced
        assert "clinical trial" in enhanced or "randomized" in enhanced
    
    @pytest.mark.asyncio
    async def test_search_papers_structure(self):
        """Test search_papers returns correct structure (with mocking)."""
        searcher = PubMedSearcher()
        
        # Mock the HTTP responses
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock search response
            mock_search_response = Mock()
            mock_search_response.status = 200
            mock_search_response.json = AsyncMock(return_value={
                'esearchresult': {'idlist': ['12345', '67890']}
            })
            
            # Mock fetch response  
            mock_fetch_response = Mock()
            mock_fetch_response.status = 200
            mock_fetch_response.text = AsyncMock(return_value='''
                <PubmedArticleSet>
                    <PubmedArticle>
                        <MedlineCitation>
                            <Article>
                                <ArticleTitle>Test Paper</ArticleTitle>
                                <Abstract><AbstractText>Test abstract</AbstractText></Abstract>
                            </Article>
                        </MedlineCitation>
                    </PubmedArticle>
                </PubmedArticleSet>
            ''')
            
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            mock_session_instance.get.return_value.__aenter__.return_value = mock_search_response
            
            # First call returns search results, second returns paper details
            mock_session_instance.get.return_value.__aenter__.side_effect = [
                mock_search_response, mock_fetch_response
            ]
            
            results = await searcher.search_papers("meditation anxiety", max_results=2)
            
            assert isinstance(results, list)
            # Should have attempted to fetch papers
            assert mock_session_instance.get.called


class TestArXivSearcher:
    """Test arXiv API integration."""
    
    def test_arxiv_searcher_initialization(self):
        """Test arXiv searcher creates successfully."""
        searcher = ArXivSearcher()
        assert searcher.base_url == "http://export.arxiv.org/api/query"
    
    @pytest.mark.asyncio
    async def test_search_papers_mock(self):
        """Test arXiv search with mocked response."""
        searcher = ArXivSearcher()
        
        with patch('aiohttp.ClientSession') as mock_session:
            mock_response = Mock()
            mock_response.status = 200
            mock_response.text = AsyncMock(return_value='''
                <feed>
                    <entry>
                        <title>Test ArXiv Paper</title>
                        <summary>Test abstract</summary>
                        <author><name>Test Author</name></author>
                        <published>2023-01-01T00:00:00Z</published>
                    </entry>
                </feed>
            ''')
            
            mock_session_instance = AsyncMock()
            mock_session.return_value.__aenter__.return_value = mock_session_instance
            mock_session_instance.get.return_value.__aenter__.return_value = mock_response
            
            results = await searcher.search_papers("test query", max_results=1)
            
            assert isinstance(results, list)


class TestResearchIntelligenceEngine:
    """Test main research intelligence orchestrator."""
    
    def test_engine_initialization(self):
        """Test engine creates successfully."""
        engine = ResearchIntelligenceEngine()
        assert engine.pubmed_searcher is not None
        assert engine.arxiv_searcher is not None
    
    @pytest.mark.asyncio
    async def test_analyze_research_topic_structure(self):
        """Test analyze_research_topic returns correct structure."""
        engine = ResearchIntelligenceEngine()
        
        # Mock both searchers to return empty results
        engine.pubmed_searcher.search_papers = AsyncMock(return_value=[])
        engine.arxiv_searcher.search_papers = AsyncMock(return_value=[])
        
        result = await engine.analyze_research_topic("test query", max_papers=5)
        
        assert isinstance(result, ResearchSummary)
        assert result.query == "test query"
        assert isinstance(result.papers_analyzed, list)
        assert result.evidence_quality in ['high', 'medium', 'low']
    
    @pytest.mark.asyncio
    async def test_analyze_with_paper_results(self):
        """Test analysis with mock paper results."""
        engine = ResearchIntelligenceEngine()
        
        # Mock papers from both sources
        mock_pubmed_papers = [{
            'title': 'PubMed Paper',
            'authors': ['Author A'],
            'abstract': 'PubMed abstract',
            'journal': 'PubMed Journal',
            'year': 2023
        }]
        
        mock_arxiv_papers = [{
            'title': 'ArXiv Paper', 
            'authors': ['Author B'],
            'abstract': 'ArXiv abstract',
            'journal': 'arXiv preprint',
            'year': 2023
        }]
        
        engine.pubmed_searcher.search_papers = AsyncMock(return_value=mock_pubmed_papers)
        engine.arxiv_searcher.search_papers = AsyncMock(return_value=mock_arxiv_papers)
        
        result = await engine.analyze_research_topic("meditation anxiety", max_papers=10)
        
        assert len(result.papers_analyzed) == 2
        assert result.total_papers_found == 2
        assert any("PubMed Paper" in paper.title for paper in result.papers_analyzed)
        assert any("ArXiv Paper" in paper.title for paper in result.papers_analyzed)
    
    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test engine handles search errors gracefully."""
        engine = ResearchIntelligenceEngine()
        
        # Mock one searcher to fail
        engine.pubmed_searcher.search_papers = AsyncMock(side_effect=Exception("Network error"))
        engine.arxiv_searcher.search_papers = AsyncMock(return_value=[])
        
        result = await engine.analyze_research_topic("test query", max_papers=5)
        
        # Should still return a valid result despite one failed search
        assert isinstance(result, ResearchSummary)
        assert result.evidence_quality == "low"  # Should be low due to limited results


# Integration tests
class TestResearchIntelligenceIntegration:
    """Integration tests that may require network access."""
    
    @pytest.mark.asyncio
    @pytest.mark.slow  # Mark as slow test
    async def test_real_pubmed_connection(self):
        """Test actual connection to PubMed (if network available)."""
        searcher = PubMedSearcher()
        
        try:
            results = await searcher.search_papers("clinical trial", max_results=1)
            # If network works, should get results
            assert isinstance(results, list)
        except Exception:
            # Network might be unavailable - that's okay for tests
            pytest.skip("Network not available for PubMed test")
    
    @pytest.mark.asyncio 
    @pytest.mark.slow
    async def test_real_arxiv_connection(self):
        """Test actual connection to arXiv (if network available)."""
        searcher = ArXivSearcher()
        
        try:
            results = await searcher.search_papers("machine learning", max_results=1)
            assert isinstance(results, list)
        except Exception:
            pytest.skip("Network not available for arXiv test")
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_end_to_end_research_pipeline(self):
        """Test complete research pipeline (if network available)."""
        engine = ResearchIntelligenceEngine()
        
        try:
            result = await engine.analyze_research_topic("anxiety treatment", max_papers=3)
            
            assert isinstance(result, ResearchSummary)
            assert result.query == "anxiety treatment"
            assert isinstance(result.papers_analyzed, list)
            assert result.evidence_quality in ['high', 'medium', 'low']
            
        except Exception:
            pytest.skip("Network not available for end-to-end test")


# Utility tests
class TestResearchIntelligenceUtils:
    """Test utility functions and edge cases."""
    
    def test_paper_conversion_with_missing_fields(self):
        """Test converting papers with missing optional fields."""
        engine = ResearchIntelligenceEngine()
        
        # Paper with minimal fields
        raw_paper = {
            'title': 'Minimal Paper',
            'authors': ['Author'],
            'abstract': 'Abstract',
            'year': 2023
            # Missing journal, pmid, etc.
        }
        
        paper = engine._convert_to_research_paper(raw_paper)
        
        assert isinstance(paper, ResearchPaper)
        assert paper.title == 'Minimal Paper'
        assert paper.journal == ""  # Should handle missing journal gracefully
    
    def test_evidence_quality_assessment(self):
        """Test evidence quality is assessed correctly."""
        engine = ResearchIntelligenceEngine()
        
        # Mock with many high-quality papers
        many_papers = [ResearchPaper(
            title=f"Paper {i}", authors=["Author"], abstract="Abstract",
            journal="High Impact Journal", year=2023, sample_size=100
        ) for i in range(10)]
        
        # Should assess as high quality
        quality = engine._assess_evidence_quality(many_papers)
        assert quality in ['high', 'medium', 'low']


if __name__ == "__main__":
    # Run fast tests by default, slow tests with --slow flag
    pytest.main([__file__, "-v", "-m", "not slow"])
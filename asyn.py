import pandas as pd
import time
from datetime import datetime
import logging
from typing import List, Dict, Any
import json
from pathlib import Path
from extract_companies_optimized import process_article, get_azure_client
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BenchmarkMetrics:
    def __init__(self):
        self.total_articles = 0
        self.processed_articles = 0
        self.failed_articles = 0
        self.total_time = 0
        self.avg_time = 0
        self.min_time = float('inf')
        self.max_time = 0
        self.articles_with_companies = 0
        self.total_companies_found = 0
        self.timings = []
        self.step_metrics = {
            "extraction": {"total": 0, "min": float('inf'), "max": 0, "count": 0, "chunks": 0},
            "search": {"total": 0, "min": float('inf'), "max": 0, "count": 0},
            "verification": {"total": 0, "min": float('inf'), "max": 0, "count": 0, "chunks": 0}
        }
        
    def add_article(self, processing_time: float, companies_found: int, success: bool, detailed_metrics: dict = None):
        self.total_articles += 1
        if success:
            self.processed_articles += 1
            self.timings.append(processing_time)
            self.total_time += processing_time
            self.min_time = min(self.min_time, processing_time)
            self.max_time = max(self.max_time, processing_time)
            if companies_found > 0:
                self.articles_with_companies += 1
                self.total_companies_found += companies_found
                
            # Add detailed step metrics
            if detailed_metrics and "step_timings" in detailed_metrics:
                for step, timing in detailed_metrics["step_timings"].items():
                    duration = timing["duration"]
                    if step in self.step_metrics:
                        self.step_metrics[step]["total"] += duration
                        self.step_metrics[step]["min"] = min(self.step_metrics[step]["min"], duration)
                        self.step_metrics[step]["max"] = max(self.step_metrics[step]["max"], duration)
                        self.step_metrics[step]["count"] += 1
                        
                        # Track number of chunks processed
                        if step in ["extraction", "verification"] and "chunks" in timing:
                            self.step_metrics[step]["chunks"] += timing["chunks"]
        else:
            self.failed_articles += 1
            
    def calculate_stats(self):
        if self.processed_articles > 0:
            self.avg_time = self.total_time / self.processed_articles
            # Calculate averages for step metrics
            for step in self.step_metrics:
                if self.step_metrics[step]["count"] > 0:
                    self.step_metrics[step]["avg"] = self.step_metrics[step]["total"] / self.step_metrics[step]["count"]
                    if "chunks" in self.step_metrics[step]:
                        self.step_metrics[step]["avg_chunks"] = self.step_metrics[step]["chunks"] / self.step_metrics[step]["count"]
            
    def get_summary(self) -> Dict[str, Any]:
        summary = {
            "total_articles": self.total_articles,
            "processed_articles": self.processed_articles,
            "failed_articles": self.failed_articles,
            "success_rate": f"{(self.processed_articles / self.total_articles * 100):.2f}%" if self.total_articles > 0 else "0%",
            "total_time": f"{self.total_time:.2f}s",
            "avg_time": f"{self.avg_time:.2f}s",
            "min_time": f"{self.min_time:.2f}s",
            "max_time": f"{self.max_time:.2f}s",
            "articles_with_companies": self.articles_with_companies,
            "total_companies_found": self.total_companies_found,
            "avg_companies_per_article": f"{(self.total_companies_found / self.processed_articles):.2f}" if self.processed_articles > 0 else "0"
        }
        
        # Add step metrics
        if self.processed_articles > 0:
            summary["step_metrics"] = {
                step: {
                    "total_time": f"{metrics['total']:.2f}s",
                    "avg_time": f"{metrics.get('avg', 0):.2f}s",
                    "min_time": f"{metrics['min']:.2f}s",
                    "max_time": f"{metrics['max']:.2f}s",
                    **({"avg_chunks": f"{metrics.get('avg_chunks', 0):.1f}"} if "chunks" in metrics else {})
                }
                for step, metrics in self.step_metrics.items()
            }
            
        return summary

def load_articles(articles_dir: str) -> List[Dict[str, str]]:
    """Load articles from a directory."""
    articles = []
    articles_path = Path(articles_dir)
    
    if not articles_path.exists():
        raise FileNotFoundError(f"Articles directory not found: {articles_dir}")
        
    for file_path in articles_path.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                articles.append({
                    "id": file_path.stem,
                    "content": content,
                    "file_path": str(file_path)
                })
        except Exception as e:
            logger.error(f"Error loading article {file_path}: {e}")
            
    return articles

async def run_benchmark_async(
    articles_dir: str,
    company_database: pd.DataFrame,
    use_cache: bool = True,
    batch_size: int = 10
) -> Dict[str, Any]:
    """Run benchmark on a set of articles with async processing."""
    metrics = BenchmarkMetrics()
    llm = get_azure_client()
    
    try:
        # Load articles
        articles = load_articles(articles_dir)
        logger.info(f"Loaded {len(articles)} articles from {articles_dir}")
        
        # Process articles in batches
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(articles) + batch_size - 1)//batch_size}")
            
            # Process batch concurrently
            tasks = [
                process_article_async(
                    article["content"],
                    company_database,
                    llm,
                    use_cache=use_cache
                )
                for article in batch
            ]
            
            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for article, (results, article_metrics) in zip(batch, batch_results):
                if isinstance(results, Exception):
                    logger.error(f"Error processing article {article['id']}: {results}")
                    metrics.add_article(0, 0, False)
                else:
                    companies_found = len(results)
                    metrics.add_article(
                        processing_time=article_metrics["execution_time"],
                        companies_found=companies_found,
                        success=True,
                        detailed_metrics=article_metrics
                    )
                    logger.info(f"Processed article {article['id']}: {article_metrics['execution_time']:.2f}s, {companies_found} companies")
            
            # Log batch progress
            logger.info(f"Completed batch {i//batch_size + 1}")
            
        # Calculate final statistics
        metrics.calculate_stats()
        return metrics.get_summary()
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return metrics.get_summary()

def run_benchmark(
    articles_dir: str,
    company_database: pd.DataFrame,
    use_cache: bool = True,
    batch_size: int = 10
) -> Dict[str, Any]:
    """Synchronous wrapper for run_benchmark_async."""
    return asyncio.run(run_benchmark_async(articles_dir, company_database, use_cache, batch_size))

def save_results(results: Dict[str, Any], output_file: str):
    """Save benchmark results to a file."""
    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to {output_file}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")

def main():
    # Configuration
    ARTICLES_DIR = "articles"  # Directory containing article text files
    COMPANY_DB_PATH = "company_database.csv"  # Path to your company database
    OUTPUT_FILE = "benchmark_results.json"
    USE_CACHE = True
    BATCH_SIZE = 10
    
    try:
        # Load company database
        company_database = pd.read_csv(COMPANY_DB_PATH)
        logger.info(f"Loaded company database with {len(company_database)} records")
        
        # Run benchmark
        results = run_benchmark(
            ARTICLES_DIR,
            company_database,
            use_cache=USE_CACHE,
            batch_size=BATCH_SIZE
        )
        
        # Save results
        save_results(results, OUTPUT_FILE)
        
        # Print summary
        logger.info("\nBenchmark Summary:")
        for key, value in results.items():
            logger.info(f"{key}: {value}")
            
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")

if __name__ == "__main__":
    main() 

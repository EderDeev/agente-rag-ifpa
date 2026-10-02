"""Modo terminal — plano B da "Blindagem do Estudante".

Roda o agente 100% local, sem servidor web, exibindo o ciclo
Thought -> Action -> Observation de cada pergunta. Ideal para a
demonstração ao vivo caso a nuvem fique indisponível.

    python -m app.cli
"""

from __future__ import annotations

import logging
import sys
import uuid

from app.agent import perguntar
from app.ingest import construir_indice, indice_existe

CORES = {"thought": "\033[36m", "action": "\033[33m", "observation": "\033[90m"}
RESET, NEGRITO, VERDE, VERMELHO = "\033[0m", "\033[1m", "\033[32m", "\033[31m"


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not indice_existe():
        print("Índice não encontrado. Indexando PDFs...")
        construir_indice()

    sessao = uuid.uuid4().hex
    total_usd = 0.0
    print(f"{NEGRITO}Assistente Acadêmico IFPA{RESET} — sessão {sessao[:8]} (digite 'sair' para encerrar)\n")

    while True:
        try:
            pergunta = input(f"{NEGRITO}Você:{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if pergunta.lower() in {"sair", "exit", "quit"}:
            break
        if not pergunta:
            continue

        r = perguntar(pergunta, sessao)
        for p in r.passos:
            rotulo = {"thought": "Thought", "action": "Action", "observation": "Observation"}[p.tipo]
            texto = p.conteudo if p.tipo != "observation" else p.conteudo[:600] + ("..." if len(p.conteudo) > 600 else "")
            print(f"{CORES[p.tipo]}{rotulo}: {texto}{RESET}")

        cor = VERDE if r.status == "respondida" else VERMELHO
        print(f"\n{NEGRITO}Assistente{RESET} [{cor}{r.status}{RESET} | confiança {r.confianca}]:\n{r.resposta}")
        for c in r.citacoes:
            print(f"  • {c.documento}, p. {c.pagina}: \"{c.trecho}\"")
        for a in r.avisos:
            print(f"  {VERMELHO}⚠ {a}{RESET}")
        if r.custo:
            c = r.custo
            total_usd += c.custo_usd
            print(
                f"\033[90m  💲 Custo: US$ {c.custo_usd:.6f} (R$ {c.custo_brl:.4f}) | {c.modelo} × {c.chamadas_llm} | "
                f"{c.tokens_entrada} in · {c.tokens_saida} out · {c.tokens_embedding} emb | "
                f"sessão: US$ {total_usd:.6f}{RESET}"
            )
        print()


if __name__ == "__main__":
    main()

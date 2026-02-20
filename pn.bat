@echo off
rem Wrapper dla ProveNuance CLI — używa venv z projektu
rem Użycie: pn docs / pn ingest ... / pn facts
"%~dp0.venv\Scripts\provenuance.exe" %*
